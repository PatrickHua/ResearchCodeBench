import os
import numpy as np
from PIL import Image
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import StableVideoDiffusionPipeline, AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel, EulerDiscreteScheduler
from diffusers.utils import export_to_video

from src.utils import *
from src.pipeline import SGI2VPipe
from src.model import MyUNet

import argparse

def read_condition(input_dir, config):
    """
    Read input condition.
    input_dir/:
        ./img.png (first frame image)
        ./traj.npy (ndarray of shape [N, (2+F), 2], where first [N, 2, 2] specifies top-left/bottom-right coordinates of bounding boxes (i.e., [[w1, h1], [w2, h2]]), while the rest of [N, F, 2] specifies trajectories of center coordinates of each bounding box across frames (in order of (w, h))

    Note: N is the number of bounding boxes placed on the first frame, F is the number of frames.
    """
    
    image_path = os.path.join(input_dir, "img.png")
    
    #Load image
    image = Image.open(image_path).convert('RGB')
    original_width, original_height = image.size
    image = image.resize((config.width, config.height))

    #load bounding_box, center_traj
    ret = np.load(os.path.join(input_dir, "traj.npy")).astype(np.float32) #N x (2+F) x 2
    ret[:,:,0] = ret[:,:,0]*config.width/original_width 
    ret[:,:,1] = ret[:,:,1]*config.height/original_height
    bounding_box = ret[:, :2].reshape(-1, 4) # N x 4
    center_traj = ret[:, 2:] #N x F x 2
    
    #Preprocess trajectory
    trajectory_points = [] # N x frames x 4 (i.e., h1, w1, h2, w2) : trajectory of bounding boxes overparameterized by top-left/bottom-right coordinates for each frame for convenience
    for j, trajectory in enumerate(center_traj):
            #For normal use
            box_traj = [] # frames x 4
            for i in range(config.num_frames):
                d = center_traj[j][i] - center_traj[j][0]
                dx, dy = d[0], d[1]
                box_traj.append(np.array([bounding_box[j][1] + dy, bounding_box[j][0] + dx, bounding_box[j][3] + dy, bounding_box[j][2] + dx], dtype=np.float32))
            trajectory_points.append(box_traj)
    return image, trajectory_points

#Approx. 4 minutes on A6000 with default config
def run(pipe, config, image, trajectory_points):
    pipe.unet.num_inference_steps = config.num_inference_steps
    pipe.unet.optimize_zero_initialize_param = True
    height, width = config.height, config.width
    motion_bucket_id = 127
    fps = 7
    num_frames = config.num_frames
    seed = config.seed
    pipe.unet.heatmap_sigma = config.heatmap_sigma
    pipe.unet.latent_fft_post_merge = config.latent_fft_post_merge
    pipe.unet.latent_fft_ratio = config.fft_ratio #range : 0.0 - 1.0
    pipe.unet.optimize_latent_iter = config.optimize_latent_iter
    pipe.unet.optimize_latent_lr = config.optimize_latent_lr
    pipe.unet.optimize_latent_time = config.optimize_latent_time
    pipe.unet.record_layer_sublayer =  config.record_layer_sublayer
    generator = torch.manual_seed(seed)
    frames = pipe(image, trajectory_points, height=height, width=width, num_frames = num_frames, decode_chunk_size=8, generator=generator, fps=fps, motion_bucket_id=motion_bucket_id, noise_aug_strength=0.02).frames[0]
    return frames

def main(config, input_dir, output_dir):

    #Path check
    assert(os.path.exists(args.input_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    
    #Read input condition
    print("Reading input condition..")
    image, trajectory_points = read_condition(input_dir, config)
    
    #Visualize
    visualize_control(image, trajectory_points=trajectory_points, save_path = os.path.join(output_dir, "condition_vis.png"))
    
    #Load pre-trained image-to-video diffusion models
    print("Loading Stable Video Diffusion..")
    svd_dir = "stabilityai/stable-video-diffusion-img2vid"
    cache_dir = "./../"
        
    feature_extractor = CLIPImageProcessor.from_pretrained(svd_dir, subfolder="feature_extractor", cache_dir=cache_dir, torch_dtype=torch.float16, variant="fp16")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(svd_dir, subfolder="vae", cache_dir=cache_dir, torch_dtype=torch.float16, variant="fp16").to("cuda")
    requires_grad(vae, False)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(svd_dir, subfolder="image_encoder", cache_dir = cache_dir, torch_dtype=torch.float16, variant="fp16").to("cuda")
    requires_grad(image_encoder, False)
    unet = MyUNet.from_pretrained(svd_dir, subfolder="unet", cache_dir=cache_dir, torch_dtype=torch.float16, variant="fp16").to("cuda")
    requires_grad(unet, False)
    scheduler = EulerDiscreteScheduler.from_pretrained(svd_dir, subfolder="scheduler", cache_dir=cache_dir, torch_dtype=torch.float16, variant="fp16")
            
    unet.inject() #inject module
    
    print("Stable Video Diffusion loaded!")
        
    #Set up pipeline
    pipe = SGI2VPipe(vae,image_encoder,unet,scheduler,feature_extractor).to(device="cuda")

    #Generate video
    frames = run(pipe, config, image, trajectory_points)
    
    #Save video
    export_to_video(frames, os.path.join(output_dir, "./result.mp4"), fps=7)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument('--input_dir', type=str, required=True, help='Path directory for input conditions.')
    parser.add_argument('--output_dir', type=str, required=True, help='Saving path directory for generated videos.')
    
    args = parser.parse_args()

    #Set up config
    class Config:
        """
        Hyperparameters
        """
        seed = 817
        height, width = 576, 1024 #resolution of generated video
        num_frames = 14
        num_inference_steps = 50 #total number of inference steps
        optimize_latent_time = list(range(30,46)) #set of timesteps to perform optimization
        optimize_latent_iter = 5 #number of optimization iterations to perform for each timestep
        optimize_latent_lr = 0.21 #learning rate for optimization
        record_layer_sublayer = [(2, 1), (2, 2)] #extract feature maps from 1st and 2nd self-attention (note: 0-indexed base) located at 2nd resolution-level of upsampling layer
        heatmap_sigma = 0.4 #standard deviation of gaussian heatmap
        fft_ratio = 0.5 #fft mix ratio
        latent_fft_post_merge = True #fft-based post-processing is enabled iff True
    
    #Run
    main(config = Config(), input_dir = args.input_dir, output_dir = args.output_dir)