import os
import numpy as np
from PIL import Image
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F
from diffusers import StableVideoDiffusionPipeline, UNetSpatioTemporalConditionModel
from diffusers.pipelines.stable_video_diffusion import StableVideoDiffusionPipelineOutput
from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionOutput
from src.utils import *

"""
Define denoising network (UNet)
"""

class MyUNet(UNetSpatioTemporalConditionModel):
    """
    Modified from SVD implementation
    https://github.com/huggingface/diffusers/blob/24c7d578baf6a8b79890101dd280278fff031d12/src/diffusers/models/unets/unet_spatio_temporal_condition.py#L32
    """
    def inject(self):
        #Replace self-attention blocks in the upsampling layers with our implementation
        for (layer, upsample_block) in enumerate(self.up_blocks):
            if layer == 0: 
                continue
            for (sublayer, trans) in enumerate(upsample_block.attentions):
                basictrans = trans.transformer_blocks[0] #BasicTransformerBlock
                basictrans.attn1.processor = self.my_self_attention(layer, sublayer)

    record_value_ = []
    
    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        added_time_ids,
        return_dict: bool = True,
    ):
        #Modified from the original implementation such that it cuts redundant computation during the optimization
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = False
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)
        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        emb = emb + aug_emb
        sample = sample.flatten(0, 1)
        emb = emb.repeat_interleave(num_frames, dim=0)
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)
        sample = self.conv_in(sample)
        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)
        down_block_res_samples = (sample,)
        for layer, downsample_block in enumerate(self.down_blocks):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )

            down_block_res_samples += res_samples
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )
        for i, upsample_block in enumerate(self.up_blocks):
            if self.training and i > max(self.record_layer_sublayer)[0]:
                return None #skip redundant computation during optimization
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    image_only_indicator=image_only_indicator,
                )
        if self.training:
            return None #skip redundant computation during optimization
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])
        if not return_dict:
            return (sample,)
        return UNetSpatioTemporalConditionOutput(sample=sample)
    
    def my_self_attention(self, layer, sublayer):
        compress_factor = [None, 4, 2, 1][layer]
        #Modified from the original implementation so that we can record semantically aligned feature maps during the optimization
        def processor(
            attn,
            hidden_states,
            encoder_hidden_states = None,
            attention_mask = None,
            temb = None,
        ):
            residual = hidden_states

            h = self.latent_shape[-2]//compress_factor
            w = self.latent_shape[-1]//compress_factor
            
            input_ndim = hidden_states.ndim
            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            query = attn.to_q(hidden_states)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            
            if self.training and ((layer, sublayer) in self.record_layer_sublayer):
                #Modified self-attention computation
                #inject key and value from the first frame to obtain semantically aligned fieature maps
                frame = query.shape[0]
                key2 = (key.reshape((1, frame)+key.shape[1:]))[:,:1].repeat((1,frame,1,1,1)).reshape(key.shape)
                value2 = (value.reshape((1, frame)+value.shape[1:]))[:,:1].repeat((1,frame,1,1,1)).reshape(value.shape)

                hidden_states = F.scaled_dot_product_attention(query, key2.clone().detach(), value2.clone().detach(), attn_mask=None, dropout_p=0.0, is_causal=False)
                hid = hidden_states.permute((0, 2, 1, 3)) #(2*batch, h*w, head, channel)
                hid = hid.reshape((hid.shape[0], h, w, -1))
                self.record_value_.append(hid)
            
            hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            hidden_states = hidden_states / attn.rescale_output_factor
            return hidden_states
        return processor

