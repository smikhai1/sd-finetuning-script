import os
import os.path as osp

from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionImg2ImgPipeline
import numpy as np
from PIL import Image
import torch
from torchvision.utils import make_grid
from tqdm import tqdm


@torch.no_grad()
def run_img2img_inference(accelerator, args, vae, unet, text_encoder, dtype,
                          init_images_dp, prompts, neg_prompts, seed, save_root_dir,
                          num_imgs_in_row=3, num_inference_steps=100, guidance_scale=8.5,
                          num_images_per_prompt=5, strength=0.5, init_img_size=512):

    vae = accelerator.unwrap_model(vae, keep_fp32_wrapper=False).to(dtype=dtype)
    if args.train_text_encoder:
        text_encoder = accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=False).to(dtype=dtype)

    unet = accelerator.unwrap_model(unet, keep_fp32_wrapper=False)
    unet = unet.to(dtype=dtype)

    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(args.pretrained_model_name_or_path,
                                                              text_encoder=text_encoder,
                                                              vae=vae,
                                                              unet=unet,
                                                              revision=args.revision,
                                                              torch_dtype=dtype)
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

    def dummy_checker(images, **kwargs):
        return images, False

    pipeline.safety_checker = dummy_checker
    init_images_w_names = load_images(init_images_dp, size=init_img_size)

    if not isinstance(prompts, (list, tuple)):
        prompts = [prompts]

    if not isinstance(neg_prompts, (list, tuple)):
        neg_prompts = [neg_prompts] * len(prompts)
    elif len(neg_prompts) == 1 and len(prompts) != 1:
        neg_prompts = neg_prompts * len(prompts)

    assert len(prompts) == len(neg_prompts)

    generator = torch.Generator(device="cuda").manual_seed(seed)

    for image, img_name in tqdm(init_images_w_names, desc='Running img2img SD pipeline ...'):
        for p_idx, prompt in enumerate(prompts):
            save_dir = osp.join(save_root_dir, prompt)
            os.makedirs(save_dir, exist_ok=True)

            neg_prompt = neg_prompts[p_idx]
            results = pipeline(prompt=prompt, image=image, generator=generator,
                               negative_prompt=neg_prompt, num_inference_steps=num_inference_steps,
                               guidance_scale=guidance_scale, num_images_per_prompt=num_images_per_prompt,
                               strength=strength).images

            grid = [torch.from_numpy(np.array(img)).permute(2, 0, 1) for img in results]
            grid.insert(0, torch.from_numpy(np.array(image)).permute(2, 0, 1))
            grid = make_grid(grid, nrow=num_imgs_in_row)
            grid = grid.permute(1, 2, 0).numpy()

            save_fp = osp.join(save_dir, osp.splitext(img_name)[0] + '.jpg')
            Image.fromarray(grid).save(save_fp, quality=70)

    del pipeline
    torch.cuda.empty_cache()

    # wrap all models back
    vae, unet = accelerator.prepare(vae, unet)
    if args.train_text_encoder:
        text_encoder = accelerator.prepare(text_encoder)

    unet.to(dtype=torch.float32).train()
    vae.to(accelerator.device, dtype=dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=dtype)
    else:
        text_encoder.to(accelerator.device, dtype=torch.float32)
        text_encoder.train()

    return text_encoder, vae, unet


def resize_image(image, size):
    w, h = image.size
    scale = size / (max(w, h))
    if h >= w:
        new_w = int(scale * w)
        new_h = size
    else:
        new_w = size
        new_h = int(scale * h)
    new_h -= new_h % 32
    new_w -= new_w % 32
    image = image.resize((new_w, new_h), resample=Image.LANCZOS)
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

