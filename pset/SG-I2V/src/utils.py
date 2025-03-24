import cv2
import os
import numpy as np
from PIL import Image
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F


"""
Define utils function
"""

"""
Define utils function
"""

def export_to_gif(
    video_frames, save_path
):
    """
    write to gif
    """
    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]
    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]
    for i in range(len(video_frames)):
        video_frames[i] = Image.fromarray(video_frames[i])
    video_frames[0].save(save_path, save_all=True, append_images=video_frames[1:], loop=0, duration=110)
    return video_frames

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
def tensor2vid(video, processor, output_type: str = "np"):
    #ref: https://github.com/huggingface/diffusers/blob/687bc2772721af584d649129f8d2a28ca56a9ad8/src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py#L61C1-L79C19
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)
        outputs.append(batch_output)
    if output_type == "np":
        outputs = np.stack(outputs)
    elif output_type == "pt":
        outputs = torch.stack(outputs)
    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil]")    
    return outputs
def visualize_control(image, trajectory_points, save_path):
    scale_factor = 1.5
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for i in range(len(trajectory_points)):
        frames = len(trajectory_points[i])
        for j in range(frames):
            source_point = trajectory_points[i][j]
            sx, sy, tx, ty = source_point[0], source_point[1], source_point[2], source_point[3]
            if j==0:
                image2 = cv2.rectangle(image.copy(), (int(sy), int(sx)), (int(ty), int(tx)), (69, 27, 255), thickness=-1)
                image = cv2.rectangle(image, (int(sy), int(sx)), (int(ty), int(tx)), (0, 0, 255), thickness=6)
                image2 = cv2.rectangle(image2, (int(sy), int(sx)), (int(ty), int(tx)), (0, 0, 255), thickness=6)
                image = cv2.addWeighted(image,0.4,image2,0.6,0)
            if j + 1 < frames:
                target_point = trajectory_points[i][j+1]
                sx2, sy2, tx2, ty2 = target_point[0], target_point[1], target_point[2], target_point[3]
                sx3 = (sx+tx)//2
                tx3 = (sx2+tx2)//2
                sy3 = (sy+ty)//2
                ty3 = (sy2+ty2)//2
                arrow_length = np.sqrt((sx3-tx3)**2 + (sy3-ty3)**2)
                green = (0,255,0)
                if j + 2 == frames:
                    image = cv2.line(image, (int(sy3), int(sx3)), (int(ty3), int(tx3)), green, thickness = int(12*scale_factor))
                    image = cv2.circle(image, (int(ty3), int(tx3)), radius = int(15*scale_factor), color = green, thickness = -1)
                    #image = cv2.arrowedLine(image, (int(sy3), int(sx3)), (int(ty3), int(tx3)), green, 12, tipLength=2) #8/arrow_length)
                else:
                    image = cv2.line(image, (int(sy3), int(sx3)), (int(ty3), int(tx3)), green, thickness = int(12*scale_factor)) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imsave(save_path, image)
    
def butterworth_low_pass_filter(latents, n=4, d_s=0.25):
    """
    Return the butterworth low pass filter mask.
    Modified from https://github.com/arthur-qiu/FreeTraj/blob/0b3ffdb932bba01ba707d689f4905c31b193468b/utils/utils_freetraj.py#L228
    """
    shape = latents.shape
    H, W = shape[-2], shape[-1]
    mask = torch.zeros_like(latents)
    if d_s==0:
        return mask
    for h in range(H):
        for w in range(W):
            d_square = ((2*h/H-1)**2 + (2*w/W-1)**2)
            mask[..., h,w] = 1 / (1 + (d_square / d_s**2)**n)
    return mask