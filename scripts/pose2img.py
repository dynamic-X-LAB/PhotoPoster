import argparse
import pickle
import os.path as osp
from omegaconf import OmegaConf
from PIL import Image
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import os, sys
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPVisionModelWithProjection

from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2img import Pose2ImagePipeline
from utils import align

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=768)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--global_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device="cuda")

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
        },
    ).to(dtype=weight_dtype, device="cuda")

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device="cuda"
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device="cuda")

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)
    generator = torch.manual_seed(args.seed)

    width, height = args.W, args.H

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )
    pose_guider.load_state_dict(
        torch.load(config.pose_guider_path, map_location="cpu"),
    )

    pipe = Pose2ImagePipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    save_dir_name = f"{time_str}--seed_{args.seed}-{args.W}x{args.H}"

    save_dir = Path(f"output/stage1/{date_str}/{save_dir_name}")
    save_dir.mkdir(exist_ok=True, parents=True)



    for idx, ref_image_path in enumerate(config['test_cases'].keys()):
        if idx%args.world_size != args.global_id: continue
        for pose_image_path in config['test_cases'][ref_image_path]:
            ref_name = osp.split(ref_image_path)[1].split('.')[0]
            pose_name = osp.split(pose_image_path)[1].split('.')[0]
            ref_image_pil = Image.open(ref_image_path).convert("RGB")
            pose_image_pil = Image.open(pose_image_path).convert("RGB")

            with open(ref_image_path+'.pose.pkl', 'rb') as f:
                ref_image_pts = [pickle.load(f)]
            with open(pose_image_path[:-4]+'.pkl', 'rb') as f:
                pose_image_pts = [pickle.load(f)]
            
            ref_image_pil, pose_images_pil = align.process(ref_image_pil, ref_image_pts, [pose_image_pil], pose_image_pts, dst_size=(width, height))
            pose_image_pil = pose_images_pil[0]

            image = pipe(
                ref_image_pil,
                pose_image_pil,
                width,
                height,
                20,
                3.5,
                generator=generator,
            ).images
            pose_image_pil = pose_image_pil.resize((width, height), resample=Image.Resampling.LANCZOS)

            image = image[0, :, 0].permute(1, 2, 0).cpu().numpy()  # (3, 512, 512)
            res_image_pil = Image.fromarray((image * 255).astype(np.uint8))
            w, h = res_image_pil.size
            canvas = Image.new("RGB", (w * 3, h), "white")
            ref_image_pil = ref_image_pil.resize((w, h))
            canvas.paste(ref_image_pil, (0, 0))
            canvas.paste(pose_image_pil, (w, 0))
            canvas.paste(res_image_pil, (w * 2, 0))

            sample_name =  f"{ref_name}__{pose_name}"
            img = canvas
            out_file = Path(
                f"{save_dir}/{sample_name}.png"
            )
            img.save(out_file)
            res_image_pil.save(f"{save_dir}/{sample_name}_res.png")
            pose_image_pil.save(f"{save_dir}/{sample_name}_pose.png")


    del pipe
    torch.cuda.empty_cache()
if __name__ == "__main__":
    main()
