import os
import os.path as osp

from diffusers import EulerAncestralDiscreteScheduler
import numpy as np
from PIL import Image
import torch
from torchvision.utils import make_grid
from tqdm import tqdm

from fixed_sd_img2img_pipeline import StableDiffusionImg2ImgPipeline


@torch.no_grad()
def run_img2img_inference(pipeline, init_images_dp, prompts, neg_prompts, seed, save_root_dir,
                          num_imgs_in_row=3, num_inference_steps=100, guidance_scale=8.5,
                          num_images_per_prompt=5, strength=0.5, init_img_size=512):

    init_images_w_names = load_images(init_images_dp, size=init_img_size)

    if not isinstance(prompts, (list, tuple)):
        prompts = [prompts]

    if not isinstance(neg_prompts, (list, tuple)):
        neg_prompts = [neg_prompts] * len(prompts)

    assert len(prompts) == len(neg_prompts)

    generator = torch.Generator(device="cuda").manual_seed(seed)

    for init_image, img_name in tqdm(init_images_w_names, desc='Running img2img SD pipeline ...'):
        for p_idx, prompt in enumerate(prompts):
            save_dir = osp.join(save_root_dir, prompt)
            os.makedirs(save_dir, exist_ok=True)

            neg_prompt = neg_prompts[p_idx]
            results = pipeline(prompt=prompt, init_image=init_image, generator=generator,
                               negative_prompt=neg_prompt, num_inference_steps=num_inference_steps,
                               guidance_scale=guidance_scale, num_images_per_prompt=num_images_per_prompt,
                               strength=strength).images

            grid = [torch.from_numpy(np.array(img)).permute(2, 0, 1) for img in results]
            grid.insert(0, torch.from_numpy(np.array(init_image)).permute(2, 0, 1))
            grid = make_grid(grid, nrow=num_imgs_in_row)
            grid = grid.permute(1, 2, 0).numpy()

            save_fp = osp.join(save_dir, osp.splitext(img_name)[0] + '.jpg')
            Image.fromarray(grid).save(save_fp, quality=70)


def make_img2img_pipeline(text_encoder, vae, unet, model_name, revision,
                          device='cuda', dtype=torch.float16):
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model_name,
                                                              text_encoder=text_encoder,
                                                              vae=vae,
                                                              unet=unet,
                                                              revision=revision,
                                                              torch_dtype=dtype)
    pipeline = pipeline.to(device)
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    return pipeline


def resize_image(image, size):
    if not isinstance(size, (tuple, list)):
        size = (size, size)
    h, w = size
    image = image.resize((w, h), resample=Image.LANCZOS)
    return image


def load_images(imgs_dir, size=512):
    images_w_names = []
    for name in os.listdir(imgs_dir):
        if name.startswith('.'):
            continue
        img = Image.open(osp.join(imgs_dir, name))
        img = resize_image(img, size)
        images_w_names.append((img, name))
    return images_w_names

