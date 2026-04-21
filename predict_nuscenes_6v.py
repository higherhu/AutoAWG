import argparse
import json
import os
from copy import deepcopy
from glob import glob
from functools import partial
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    CogVideoXDDIMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
)
from einops import rearrange
from transformers import T5EncoderModel
from controlnet_aux import (
    PidiNetDetector, LineartStandardDetector, LeresDetector
)
from ultralytics import YOLO
import numpy as np
from safetensors.torch import load_file

from cogvideox.models.autoencoder_magvit import AutoencoderKLCogVideoX
from cogvideox.models.transformer3d import CogVideoXTransformer3DModel
from cogvideox.pipeline.pipeline_cogvideox_inpaint import (
    CogVideoX_Fun_Pipeline_Inpaint_6v
)
from cogvideox.utils.utils import (
    save_videos_grid, cal_lineart_standard, cal_pidi_online,
    cal_leres_online, cal_detect_online,
    load_deeplabv3plus_model, cal_mask_online
)


INPUT_DATA = [
    {
        "img_dir": "data/nuScenes_samples/n008-2018-05-21-11-06-59-0400_000001",
        "raw_weather_cap": "driving in the sunny weather"
    },
    {
        "img_dir": "data/nuScenes_samples/n015-2018-09-27-15-33-17+0800_000100",
        "raw_weather_cap": "driving in the sunny weather"
    }
]


def load_patch_safetensors(path):
    list_tensors = glob(path+"/*.safetensors")
    all={}
    for x in list_tensors:
        tmp = load_file(x)
        all.update(tmp)
    return all


def get_segments(video_length, sub_video_length):
    on_overlap_length = sub_video_length - 1
    seg_num = (video_length - 1) // on_overlap_length
    pairs = []
    for i in range(seg_num):
        pairs.append([i * on_overlap_length, (i + 1) * on_overlap_length + 1])

    return pairs


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Weather Transfer Predict Demo", add_help=True)
    parser.add_argument("--transformer_path", type=str, default='./checkpoints/adverse_weather/transformer/diffusion_pytorch_model.safetensors')
    parser.add_argument("--fps", type=int, default=10, required=False)
    parser.add_argument("--save_dir", type=str, default="output_nuscene_6v", required=False)
    parser.add_argument("--num_workers", type=int, default=1, required=False)
    parser.add_argument("--current_worker", type=int, default=0, required=False)
    parser.add_argument("--max_frames", type=int, default=45, required=False)
    parser.add_argument(
        "--strong_control",
        default="lineart_standard",
        help="Whether to combine sketch, seg and canny to a 3-channel img or not."
    )
    parser.add_argument(
        "--weak_control",
        default="pidi",
        help="Whether to combine sketch, seg and canny to a 3-channel img or not."
    )
    parser.add_argument(
        "--third_control",
        default='leres',
        help="Whether to use a third control or not."
    )


    args = parser.parse_args()
    print(args)

    low_gpu_memory_mode = False

    num_workers = args.num_workers
    current_worker = args.current_worker
    # Config and model path

    select_type = ["fuse"]

    assert len(select_type) == 1, f"{args.transformer_path} is not a valid path"
    args.fuse_control_type = select_type[0]

    print(f'fuse type: {args.fuse_control_type}')

    from datetime import datetime
    current_fime = datetime.now().strftime("%Y%m%d")

    model_name          = "checkpoints/alibaba-pai/CogVideoX-Fun-V1.5-5b-InP/"
    transformer_path         = args.transformer_path
    save_root               = args.save_dir
    in_channels = 33 + ((16 * 3) if args.fuse_control_type == 'seperate' else 16)
    num_inference_steps  = 20
    video_length        = 45

    row = 2
    col = 3

    # Choose the sampler in "Euler" "Euler A" "DPM++" "PNDM" "DDIM_Cog" and "DDIM_Origin"
    sampler_name        = "DDIM_Origin"

    # Load pretrained model if need
    vae_path            = None
    lora_path           = None

    # Other params
    fps                 = args.fps

    # If you want to generate ultra long videos, please set partial_video_length as the length of each sub video segment
    partial_video_length = None
    overlap_video_length = 4

    # Use torch.float16 if GPU does not support torch.bfloat16
    # ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
    weight_dtype            = torch.bfloat16
    # If you want to generate from text, please set the validation_image_start = None and validation_image_end = None
    # validation_image_start  = "asset/1.png"
    validation_image_end    = None

    # prompts
    # prompt                  = "driving scene"
    negative_prompt         = "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion. "
    guidance_scale          = 6.0
    seed                    = 43
    strength = 1.0

    lora_weight             = 0.55

    transformer = CogVideoXTransformer3DModel.from_pretrained_2d(
        model_name,
        subfolder="transformer",
    ).to(weight_dtype)

    if transformer_path is not None:
        print(f"From checkpoint: {transformer_path}")
        state_dict = load_file(transformer_path)
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        proj_ori = deepcopy(transformer.patch_embed.proj)
        patch_size_t = transformer.patch_embed.patch_size_t
        patch_size = transformer.patch_embed.patch_size
        proj_ori = transformer.patch_embed.proj
        transformer.patch_embed.proj = nn.Linear(
        in_channels * patch_size_t * patch_size * patch_size,
        transformer.patch_embed.embed_dim,
        ).to(weight_dtype)
        m, u = transformer.load_state_dict(state_dict, strict=True)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    pidinet = PidiNetDetector.from_pretrained("lllyasviel/Annotators").netNetwork
    pidinet = pidinet.to("cuda")
    pidinet.eval()
    leresnet = LeresDetector.from_pretrained("lllyasviel/Annotators").model
    leresnet.to("cuda")
    # leresnet.eval()  # !!! cannot add this line!
    lineart_standard = LineartStandardDetector()
    yolo_model = YOLO("checkpoints/yolo11x-seg.pt")
    yolo_model.to("cuda")
    yolo_model.eval()
    sky_seg_model = load_deeplabv3plus_model("checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar")
    sky_seg_model.to('cuda')
    sky_seg_model.eval()

    controlfunc_map = {
        "pidi": partial(cal_pidi_online, pidi_model=pidinet),
        "lineart_standard": partial(cal_lineart_standard, lineart_standard_model=None),
        "leres": partial(cal_leres_online, leres_model=leresnet),
    }
    func_cstrong = controlfunc_map[args.strong_control]
    func_cweak = controlfunc_map[args.weak_control]
    if args.third_control is not None:
        func_third = controlfunc_map[args.third_control]
    print(f"Strong control: {args.strong_control}, Weak control: {args.weak_control}, Third control: {args.third_control}")

    # Get Vae
    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_name,
        subfolder="vae"
    ).to(weight_dtype)

    if vae_path is not None:
        print(f"From checkpoint: {vae_path}")
        if vae_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(vae_path)
        else:
            state_dict = torch.load(vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    @torch.no_grad()
    def _slice_vae_views(pixel_values, row=2, col=3):
        pixel_values = rearrange(pixel_values, "b f c (m h) (n w) -> (m n b) c f h w", m=row, n=col)
        bs = 1
        new_pixel_values = []
        for i in range(0, pixel_values.shape[0], bs):
            pixel_values_bs = pixel_values[i : i + bs]
            pixel_values_bs = vae.encode(pixel_values_bs)[0]
            pixel_values_bs = pixel_values_bs.sample()
            new_pixel_values.append(pixel_values_bs)

        new_pixel_values = torch.cat(new_pixel_values, dim = 0)
        new_pixel_values = rearrange(new_pixel_values, "(m n b) c f h w -> b c f (m h) (n w)", m=row, n=col)
        return new_pixel_values

    text_encoder = T5EncoderModel.from_pretrained(
        model_name, subfolder="text_encoder", torch_dtype=weight_dtype
    )
    # Get Scheduler
    Choosen_Scheduler = scheduler_dict = {
        "Euler": EulerDiscreteScheduler,
        "Euler A": EulerAncestralDiscreteScheduler,
        "DPM++": DPMSolverMultistepScheduler,
        "PNDM": PNDMScheduler,
        "DDIM_Cog": CogVideoXDDIMScheduler,
        "DDIM_Origin": DDIMScheduler,
    }[sampler_name]
    scheduler = Choosen_Scheduler.from_pretrained(
        model_name,
        subfolder="scheduler"
    )

    if transformer.config.in_channels != vae.config.latent_channels:
        pipeline = CogVideoX_Fun_Pipeline_Inpaint_6v.from_pretrained(
            model_name,
            vae=vae,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=weight_dtype
        )
    else:
        raise ValueError("in_channels of transformer and vae must be the same")

    if low_gpu_memory_mode:
        pipeline.enable_sequential_cpu_offload()
    else:
        pipeline.enable_model_cpu_offload()

    generator = torch.Generator(device="cuda").manual_seed(seed)

    all_weather = ['snowy', 'foggy', 'night', 'rainy', "same"]
    #all_weather = ['snowy', "same"]

    video_items = [{'img_dir': x['img_dir'], "raw_weather_cap": x["raw_weather_cap"]} for x in INPUT_DATA]

    sub_w, sub_h = 960, 544
    video_h, video_w = 544 * 2, 960 * 3

    sub_width_list = [sub_w, sub_w, sub_w]
    sub_height_list = [sub_h, sub_h]
    last_gen_img = None
    with torch.no_grad():
        pairs = []
        for item in video_items:
            for weather in all_weather:
                if weather == 'night':
                    prompt = 'driving in the night'
                elif weather == "same":
                    prompt = item["raw_weather_cap"]
                else:
                    prompt = f'driving in the {weather} weather'

                pairs.append((item, weather, prompt))

        print(f"all_pairs before: {len(pairs)}")
        print(f"current_worker: {current_worker}, num_workers: {num_workers}")
        pairs = pairs[current_worker::num_workers]
        print(f"all_pairs before: {len(pairs)}")

        # print(f'all_pairs: {pairs}')
        for pair in pairs:
            item, weather, prompt = pair
            print(f"img_dir: {item['img_dir']}, Weather: {weather}, prompt: {prompt}")

            img_dir = item['img_dir']
            prefix = os.path.basename(img_dir)
            save_path = os.path.join(save_root, prefix)
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, prefix + ".gen.mp4")

            gen_path = save_path.replace(".gen.mp4", f"_{weather}.gen.mp4")
            if os.path.exists(gen_path):
                print(f'{gen_path} exists, skip')
                continue

            print(f'run generate {gen_path}...')
            img_list = glob(os.path.join(img_dir, "*.jpg"))
            if len(img_list) < video_length:
                print(f"frames num is {len(img_list)} less than video_length {video_length}, skip")
            img_list.sort()
            img_list = img_list
            img_list = [cv2.imread(img_path) for img_path in img_list]
            img_list = [cv2.resize(img, (video_w, video_h)) for img in img_list]
            img_list = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_list]

            segment = get_segments(min(len(img_list), args.max_frames), video_length)
            print(f"segment: {segment}")

            out = []
            for seg_index, (seg_start, seg_end) in enumerate(segment):
                print(f"Segment index: {seg_index}")

                pixel_values = np.array(img_list[seg_start: seg_end])

                origin_video = [torch.from_numpy(pixel_values.copy()).permute(0, 3, 1, 2).contiguous()]
                origin_video = torch.cat(origin_video, dim = -1).to("cuda").to(weight_dtype)
                origin_video = torch.unsqueeze(origin_video, dim=0)

                # b c f h w
                pixel_values = (torch.from_numpy(pixel_values).permute(3, 0, 1, 2).contiguous())
                pixel_values = pixel_values / 255.  # 归一化到[0,1]
                pixel_values = torch.unsqueeze(pixel_values, dim=0)
                pixel_values = pixel_values.to("cuda").to(weight_dtype)

                pixel_values_0_255_tmp = rearrange(origin_video, "b c f (m h) (n w) -> (m n b) c f h w", m=row, n=col)
                control_strong = func_cstrong(pixel_values=pixel_values_0_255_tmp, width_list=sub_width_list[:1]).to('cuda', dtype=weight_dtype)
                control_weak = func_cweak(pixel_values=pixel_values_0_255_tmp, width_list=sub_width_list[:1]).to('cuda', dtype=weight_dtype)
                control_third = func_third(pixel_values=pixel_values_0_255_tmp, width_list=sub_width_list[:1]).to('cuda', dtype=weight_dtype)
                sky_mask = cal_mask_online(sky_seg_model, pixel_values=pixel_values_0_255_tmp, width_list=sub_width_list[:1]).to("cuda").to(dtype=weight_dtype)
                yolo_mask = cal_detect_online(yolo_model, pixel_values=pixel_values_0_255_tmp, width_list=sub_width_list[:1]).to("cuda").to(dtype=weight_dtype)

                control_strong = rearrange(control_strong, "(m n b) c f h w -> b c f (m h) (n w)", m=row, n=col)
                control_weak = rearrange(control_weak, "(m n b) c f h w -> b c f (m h) (n w)", m=row, n=col)
                control_third = rearrange(control_third, "(m n b) c f h w -> b c f (m h) (n w)", m=row, n=col)
                sky_mask = rearrange(sky_mask, "(m n b) c f h w -> b c f (m h) (n w)", m=row, n=col)
                yolo_mask = rearrange(yolo_mask, "(m n b) c f h w -> b c f (m h) (n w)", m=row, n=col)

                seg = yolo_mask
                if args.fuse_control_type == 'fuse':
                    # lineart_standard
                    control_strong = torch.where(torch.logical_and(seg > 0, seg < 30), control_strong, 0)
                    # pidi
                    control_weak = torch.where(sky_mask > 0, 0, control_weak)
                    # leres
                    control_third = control_third
                    control_vid = torch.cat([control_third, control_strong, control_weak], dim=2)
                elif args.fuse_control_type == 'concat':
                    control_vid = torch.cat([control_third, control_strong, control_weak], dim=2)
                elif args.fuse_control_type == 'seperate':
                    control_vid = []
                    control_vid.append(torch.tile(control_third, (1 , 1, 3, 1, 1)))
                    control_vid.append(torch.tile(control_strong, (1 , 1, 3, 1, 1)))
                    control_vid.append(torch.tile(control_weak, (1 , 1, 3, 1, 1)))
                else:
                    assert False, "fuse_control_type must be fuse or concat or seperate"

                os.makedirs(os.path.dirname(save_path), exist_ok=True)


                if isinstance(control_vid, list):
                    tmp_control_latents_list = []
                    for control_vid_single in control_vid:
                        tmp_control_latents_list.append(_slice_vae_views(control_vid_single))
                    control_latent = torch.cat(tmp_control_latents_list, dim=1)
                else:
                    control_latent = _slice_vae_views(control_vid)

                #concat sketch and seg
                control_latent = rearrange(control_latent, "B C T H W -> B T C H W")

                print(f"control_latent.shape: {control_latent.shape}, control_latent.shape: {control_latent.shape}")

                B, T, C, H, W = seg.shape
                input_video_mask = torch.ones((B, 1, video_length, H, W), dtype=pixel_values.dtype, device='cuda')

                if seg_index == 0:
                    print('t2v mode')
                else:
                    print('i2v mode')
                    input_video_mask[:, :, 0] = 0

                    pixel_values[:, :, 0] = last_gen_img


                print('pixel_values shape: ', pixel_values.shape)
                print('control_latent shape: ', control_latent.shape)
                print('input_video_mask shape: ', input_video_mask.shape)

                with torch.no_grad():
                    sample = pipeline(
                        prompt,
                        num_frames = video_length,
                        negative_prompt = negative_prompt,
                        height      = video_h,
                        width       = video_w,
                        generator   = generator,
                        guidance_scale = guidance_scale,
                        num_inference_steps = num_inference_steps,
                        video = pixel_values, # b c f h w
                        seg_embed = control_latent, # b f c h w
                        mask_video = input_video_mask, # b c f h w
                        strength = strength,
                    ).videos

                    if seg_index == 0:
                        out.append(sample)
                    else:
                        out.append(sample[:,:,1:video_length])
                    # update the last frame
                    last_gen_img=sample[:,:,-1]

            out_video=torch.cat(out,dim=2)

            save_videos_grid(out_video, gen_path, fps=fps, save_grid=False)
            print(f"save to {save_path}")
