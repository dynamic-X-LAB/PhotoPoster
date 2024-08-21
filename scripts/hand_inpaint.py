import argparse
from pathlib import Path
from datetime import datetime
import os
import os.path as osp
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageFilter
from transformers import CLIPTextModel, CLIPTokenizer,  AutoTokenizer
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)

from src.pipelines.pipeline_controlnet_inpaint import StableDiffusionControlNetInpaintPipeline
from src.models.controlnet import ControlNetModel
from utils.pose_utils import crop_hand, combine_images, pil2pose, get_hand_meta

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--global_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=15)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    generator = torch.manual_seed(args.seed)
    config = OmegaConf.load(args.config)
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    pretrained_model_name_or_path = config.pretrained_base_model_path
    model_root = config.controlnet_path
    crop_ratio = config.crop_ratio
    control_scale = config.control_scale
    guidance_scale = config.guidance_scale
    strength = config.strength
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")

    save_dir_name = f"{time_str}"
    save_dir = Path(f"output/hand_inpaint/{date_str}/{save_dir_name}-crop{crop_ratio}-control{control_scale}-guidescale{guidance_scale}-strength{strength}")
    save_dir.mkdir(exist_ok=True, parents=True)

    prompt = 'Clear human hands'
    neg_prompt = 'longbody, bad hands, missing fingers, extra digit, fewer digits, black hands, long fingers'

    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder").to(dtype=weight_dtype, device="cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae").to(dtype=weight_dtype, device="cuda")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet").to(dtype=weight_dtype, device="cuda")
    controlnet = ControlNetModel.from_pretrained(model_root, torch_dtype=torch.float32, use_safetensors=True).to(dtype=weight_dtype, device="cuda")
    pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=weight_dtype,
    ).to("cuda")
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    
    for idx, image_path in enumerate(config['test_cases'].keys()):
        if idx%args.world_size != args.global_id: continue
        image_pil = Image.open(image_path)
        pic_name = osp.split(image_path)[1].split('.')[0]
        res_image = np.array(image_pil)
        _, res_meta = pil2pose(image_pil)
        res_pose_pil = Image.open(config['test_cases'][image_path][0])

        cache1 = []
        cache2 = [image_pil, res_pose_pil]

        for hidx in [0,1]:
            w,h = res_meta['W'], res_meta['H']
            hands = res_meta['hands']
            hands_score = res_meta['hands_score']
            hand_bbox1 = get_hand_meta(hands, hands_score, w, h, hidx=hidx, margin=0.2, conf=7.8)
            if not hand_bbox1: continue
            res_crop, res_guider, res_hand_vis_flag, res_new_hand_bbox, res_crop_bbox = crop_hand(Image.fromarray(res_image), res_meta, hidx=hidx, crop_ratio=crop_ratio, new_size=384,margin=0.3)
            x1,y1,x2,y2 = res_crop_bbox
            res_pose_crop = Image.fromarray(np.array(res_pose_pil)[y1:y2, x1:x2, :]).resize((384, 384))

            ref_kpt = res_meta['bodies.candidate2'][0]
            if x1<ref_kpt[0]<x2 and y1<ref_kpt[1]<y2:
                ref_kpt = res_meta['bodies.candidate2'][16] if hidx==0 else res_meta['bodies.candidate2'][17]
                ref_size = max(x2-x1, y2-y1)//5
            else:
                ref_size = max(x2-x1, y2-y1)//2
            x1,y1,x2,y2 = int(max(ref_kpt[0]-ref_size, 0)), int(max(ref_kpt[1]-ref_size, 0)), int(min(ref_kpt[0]+ref_size, w)) , int(min(ref_kpt[1]+ref_size, h))
            head_pic = Image.fromarray(res_image[y1:y2, x1:x2, :]).resize((384,384)).filter(ImageFilter.GaussianBlur(radius=15))

            x1,y1,x2,y2 = res_new_hand_bbox
            x1 = int(max(x1, 0))
            x2 = int(min(x2, np.array(res_crop).shape[0]))
            y1 = int(max(y1, 0))
            y2 = int(min(y2, np.array(res_crop).shape[1]))
            mask = np.zeros_like(res_guider, dtype='uint8')
            mask[y1:y2, x1:x2] = 255
            mask = Image.fromarray(mask)

            with torch.autocast("cuda"):
                res = pipeline(prompt, res_guider, mask, res_pose_crop, head_pic, 
                 num_inference_steps=50, controlnet_conditioning_scale=control_scale, generator=generator, 
                 negative_prompt=neg_prompt, guidance_scale=guidance_scale, strength=strength).images[0]
            x1,y1,x2,y2 = res_crop_bbox
            res_image[y1:y2,x1:x2,:] = np.array(res.resize((x2-x1, y2-y1)))
            cache1 += [res_pose_crop, mask, res_crop, head_pic, res]
        cache2.append(Image.fromarray(res_image))

        if len(cache1)>0:
            out_image1 = combine_images(cache1, rownum=5)
            out_image1.save(f"{save_dir}/{pic_name}_hand.png")
        if len(cache2)>2:
            out_image2 = combine_images(cache2, rownum=3)
            out_image2.save(f"{save_dir}/{pic_name}_diff.png")
        out_image_pil = Image.fromarray(res_image)
        out_image_pil.save(f"{save_dir}/{pic_name}.png")

