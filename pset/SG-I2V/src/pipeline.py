import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from diffusers import StableVideoDiffusionPipeline
from diffusers.pipelines.stable_video_diffusion import StableVideoDiffusionPipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _append_dims
from src.utils import *


"""
Define Pipeline
"""

class SGI2VPipe(StableVideoDiffusionPipeline):
    """
    Modified from the original SVD pipeline
    ref: https://github.com/huggingface/diffusers/blob/24c7d578baf6a8b79890101dd280278fff031d12/src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py#L139
    """
    heatmap = {}
    def get_gaussian_heatmap(self, h, w):
        """
        Generate gaussian heatmap
        Modified from https://github.com/showlab/DragAnything/blob/main/demo.py#L380
        """
        if (h,w) in self.heatmap:
            isotropicGrayscaleImage = self.heatmap[(h,w)]
        else:
            sigy = self.unet.heatmap_sigma*(h/2)
            sigx = self.unet.heatmap_sigma*(w/2)

            cx = w/2
            cy = h/2
            isotropicGrayscaleImage = np.zeros((h, w), np.float32)
            for y in range(h):
                for x in range(w):
                    isotropicGrayscaleImage[y, x] = 1 / 2 / np.pi / (sigx*sigy) * np.exp(
                        -1 / 2 * ((x+0.5 - cx) ** 2 / (sigx ** 2) + (y+0.5 - cy) ** 2 / (sigy ** 2)))
            isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage)).astype(np.float32)
            self.heatmap[(h,w)] = isotropicGrayscaleImage
        return torch.from_numpy(isotropicGrayscaleImage).cuda()
    
    def optimize_latent(self, latents, trajectory_points, t, image_latents, image_embeddings, added_time_ids):
        """
        trajectory_points : N x frames x 4 (i.e. upper-left and bottom-right corners of bounding box : sx,sy,tx,ty)
        """
        original_latents = latents.clone().detach()
        
        if self.unet.optimize_latent_iter > 0: 
            self.unet = self.unet.to(dtype=torch.float32)
            self.unet.enable_gradient_checkpointing()
            self.unet.train(True)
            latents = latents.to(dtype=torch.float32)
            image_latents = image_latents.to(dtype=torch.float32)
            image_embeddings = image_embeddings.to(dtype=torch.float32)
            added_time_ids = added_time_ids.to(dtype=torch.float32)
            t = t.to(dtype=torch.float32)
            
            with torch.enable_grad():
                latents = latents.clone().detach().requires_grad_(True)        
                optimizer = None 
                scaler = torch.cuda.amp.GradScaler()
                target_features = [None]*len(trajectory_points)
                
                for iter in range(self.unet.optimize_latent_iter):
                    
                    latent_model_input = self.scheduler.scale_model_input(latents, t)
                    latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                    self.unet.record_value_ = []
                    with torch.autocast(device_type = "cuda", dtype=torch.float16):
                        self.unet(latent_model_input,t,encoder_hidden_states=image_embeddings,added_time_ids=added_time_ids,return_dict=False)
                    
                    features = [] #list([frame, h, w, feature])

                    #Upsample recorded feature maps to the latent size
                    h, w = (self.unet.latent_shape[-2], self.unet.latent_shape[-1])
                    for i in range(len(self.unet.record_value_)):
                        fet = self.unet.record_value_[i].to(dtype=torch.float32).permute((0,3,1,2)) #[frame, feature, h, w]
                        fet = F.interpolate(fet, size=(h,w), mode="bilinear")
                        features.append(fet.permute((0,2,3,1)))

                    feature = torch.cat(features, dim=-1) #[frames, h, w, features]
                    
                    self.unet.record_value_ = []
                        
                    compress_factor = 8*self.unet.latent_shape[-2]//h
                        
                    frames = features[0].shape[0]
                        
                    loss = 0
                    loss_cnt = 0
                    
                    for j in range(frames):
                        #iterate over each control point
                        for point_idx in range(len(trajectory_points)):  
                            cur_point = (trajectory_points[point_idx][j]//compress_factor).astype(np.int32)
                            
                            sx, sy = cur_point[0], cur_point[1]
                            tx, ty = max(sx+1, cur_point[2]), max(sy+1, cur_point[3])
                            
                            #boundary check
                            sx_, sy_, tx_, ty_ = max(sx, 0), max(sy, 0), min(tx, feature.shape[1]), min(ty, feature.shape[2])
                            
                            #compute offset
                            osx, osy, otx, oty  = sx_ - sx, sy_ - sy, tx_ - sx, ty_ - sy

                            if sx_ >= tx_ or sy_ >= ty_:
                                #trajectory point goes beyond the image boundary
                                if j==0:
                                    print("Invalid trajectory, the initial boundaing box should not go beyond image boundary!!")
                                    exit(1)
                                continue
                            
                            if j == 0:
                                #Record feature maps of the first frame
                                target_features[point_idx] = feature[0,sx:tx,sy:ty].clone().detach().requires_grad_(False)
                            
                            #Compute loss
                            if j > 0:
                                    target = target_features[point_idx].unsqueeze(0) #[1, h, w, feature]
                                    target = F.interpolate(target.permute((0,3,1,2)),size=(tx-sx, ty-sy), mode="bilinear") #[1,feature,h,w]
                                    target = target.permute((0,2,3,1))[0] #[h, w, feature]
                                    target = target[osx:otx, osy:oty] #[h', w', feature]
                                    source = feature[j,sx_:tx_,sy_:ty_]
                                    
                                    #compute pixel-wise difference
                                    pixel_wise_loss = F.mse_loss(target, source, reduction="none").mean(dim=-1)
                                    
                                    #gaussian weight applied
                                    mask = self.get_gaussian_heatmap(tx-sx, ty-sy)
                                    mask = mask[osx:otx, osy:oty]
                                    assert(mask.shape == pixel_wise_loss.shape)

                                    #weight loss depending on the weight
                                    pixel_wise_loss = pixel_wise_loss
                                    
                                    #add up to the loss
                                    loss = loss + (mask*pixel_wise_loss).sum()
                                    loss_cnt += mask.sum()
                        
                    loss = loss/max(1e-8, loss_cnt)
                    
                    if optimizer == None:
                        #Initialize optimizer
                        optimizer = torch.optim.AdamW([latents], lr = self.unet.optimize_latent_lr) 
                    
                    self.unet.zero_grad()
                    optimizer.zero_grad()
                    if loss_cnt > 0:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                    del loss
                    del feature
                    del self.unet.record_value_
                    torch.cuda.empty_cache()

                    if loss_cnt==0:
                        break #nothing to optimize
            
            if self.unet.latent_fft_post_merge:
                fft_axis = (-2, -1) #H and W
                with torch.no_grad():
                    #latents: [1, frames = 14, channel = 4, h, w]               
                    #create low pass filter
                    LPF = butterworth_low_pass_filter(latents, d_s = self.unet.latent_fft_ratio)
                    
                    #FFT
                    latents_freq = torch.fft.fftn(latents, dim=fft_axis)
                    latents_freq = torch.fft.fftshift(latents_freq, dim=fft_axis)
                    original_latents_freq = torch.fft.fftn(original_latents.to(dtype=torch.float32), dim=fft_axis)
                    original_latents_freq = torch.fft.fftshift(original_latents_freq, dim=fft_axis)
        
                    #frequency mix
                    HPF = 1 - LPF
                    new_freq = latents_freq*LPF + original_latents_freq*HPF
        
                    #IFFT
                    new_freq = torch.fft.ifftshift(new_freq, dim=fft_axis)
                    latents = torch.fft.ifftn(new_freq, dim=fft_axis).real
            
            self.unet = self.unet.to(dtype=torch.float16)
            latents = latents.to(dtype=torch.float16)
            image_latents = image_latents.to(dtype=torch.float16)
            image_embeddings = image_embeddings.to(dtype=torch.float16)
            added_time_ids = added_time_ids.to(dtype=torch.float16)
            t = t.to(dtype=torch.float16)
            self.unet.train(False)
        
        latents = latents.detach().requires_grad_(False)
        return latents
    
    def __call__(self, image, trajectory_points, height, width, num_frames, min_guidance_scale = 1.0, max_guidance_scale = 3.0, fps = 7,
                 generator = None, motion_bucket_id = 127, noise_aug_strength = 0.02, decode_chunk_size = 8):
        #Modified from the original implementaion such that the pipeline incorporates our latent optimization procedure
        batch_size = 1
        fps = fps - 1
        self._guidance_scale = max_guidance_scale
        image_embeddings = self._encode_image(image, "cuda", 1, self.do_classifier_free_guidance)
        try: 
            image = self.image_processor.preprocess(image, height=height, width=width).to("cuda")
        except:
            self.image_processor = self.video_processor 
            image = self.image_processor.preprocess(image, height=height, width=width).to("cuda")
        noise = randn_tensor(image.shape, generator=generator, device=image.device, dtype=image.dtype).to("cuda")
        image = image + noise_aug_strength * noise
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)
        image_latents = self._encode_vae_image(
            image,
            device="cuda",
            num_videos_per_prompt=1,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        image_latents = image_latents.to(image_embeddings.dtype)
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            1,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to("cuda")
        self.scheduler.set_timesteps(self.unet.num_inference_steps, device="cuda")
        timesteps = self.scheduler.timesteps
        num_channels_latents = self.unet.config.in_channels
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        latents = randn_tensor(shape, generator=generator, device=image.device, dtype=image_embeddings.dtype).to("cuda")
        latents = latents * self.scheduler.init_noise_sigma # scale the initial noise by the standard deviation required by the scheduler
        self.unet.latent_shape = latents.shape
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to("cuda", latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)
        self._guidance_scale = guidance_scale

        #Free cuda memory
        self.vae = self.vae.to("cpu")
        self.image_encoder = self.image_encoder.to("cpu")
        torch.cuda.empty_cache()
            
        #Denoising loop
        num_warmup_steps = len(timesteps) - self.unet.num_inference_steps * self.scheduler.order #num_warmup_steps = 0 in our setting
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=self.unet.num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                self.unet.cur_timestep = len(timesteps) - i
                
                if (self.unet.cur_timestep in self.unet.optimize_latent_time):
                    #update latent through optimization
                    latents = self.optimize_latent(latents, trajectory_points, t, image_latents[1:], image_embeddings[1:], added_time_ids[1:])
                
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=image_embeddings,
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents = self.scheduler.step(noise_pred, t, latents,  s_churn = 0.0).prev_sample
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        self.vae = self.vae.to("cuda")
        self.image_encoder = self.image_encoder.to("cuda")
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
        frames = self.decode_latents(latents, num_frames, decode_chunk_size)
        frames = tensor2vid(frames, self.image_processor, output_type="np")
        self.maybe_free_model_hooks()
        return StableVideoDiffusionPipelineOutput(frames=frames)