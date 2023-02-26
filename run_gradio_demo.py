from argparse import ArgumentParser

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
import gradio as gr
import torch
from PIL import Image


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--model_id', type=str, required=True,
                        help='Model ID from HF Hub or local path to the model')
    parser.add_argument('--access_token', type=str, required=True,
                        help='HF token')
    parser.add_argument('--prefix', type=str, default='',
                        help='Prompt prefix')
    parser.add_argument('--revision', type=str, default=None,
                        help='Revision of the model')

    return parser.parse_args()


def error_str(error, title="Error"):
    return f"""#### {title}
            {error}""" if error else ""


def inference(pipe, pipe_i2i, prompt, guidance, steps, width=640, height=640, seed=0, img=None, strength=0.5, neg_prompt="",
              auto_prefix=False, prefix=''):
    generator = torch.Generator('cuda').manual_seed(seed) if seed != 0 else None
    prompt = f"{prefix} {prompt}" if auto_prefix else prompt

    try:
        if img is not None:
            return img_to_img(pipe_i2i, prompt, neg_prompt, img, strength, guidance, steps, width, height, generator), None
        else:
            return txt_to_img(pipe, prompt, neg_prompt, guidance, steps, width, height, generator), None
    except Exception as e:
        return None, error_str(e)


def txt_to_img(pipeline, prompt, neg_prompt, guidance, steps, width, height, generator):
    result = pipeline(
        prompt,
        negative_prompt=neg_prompt,
        num_inference_steps=int(steps),
        guidance_scale=guidance,
        width=width,
        height=height,
        generator=generator)

    return result.images[0]


def img_to_img(pipeline, prompt, neg_prompt, img, strength, guidance, steps, width, height, generator):
    ratio = min(height / img.height, width / img.width)
    img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    result = pipeline(
        prompt,
        negative_prompt=neg_prompt,
        init_image=img,
        num_inference_steps=int(steps),
        strength=strength,
        guidance_scale=guidance,
        generator=generator)

    return result.images[0]


def main(args):
    model_id = args.model_id
    access_token = args.access_token
    revision = args.revision
    prefix = args.prefix

    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler",
                                                                use_auth_token=access_token, revision=revision)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        scheduler=scheduler,
        use_auth_token=access_token,
        revision=revision)

    pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        scheduler=scheduler,
        use_auth_token=access_token,
        revision=revision
    )

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        pipe_i2i = pipe_i2i.to("cuda")


    css = """.main-div div{display:inline-flex;align-items:center;gap:.8rem;font-size:1.75rem}.main-div div h1{font-weight:900;margin-bottom:7px}.main-div p{margin-bottom:10px;font-size:94%}a{text-decoration:underline}.tabs{margin-top:0;margin-bottom:0}#gallery{min-height:20rem}
    """
    with gr.Blocks(css=css) as demo:
        gr.HTML(
            f"""
                <div class="main-div">
                  <div>
                    <h1>22h Diffusion v0.1</h1>
                  </div>
                  <p>
                   Demo for <a href="https://huggingface.co/22h/vintedois-diffusion-v0-1">22h Diffusion v0-1</a> Stable Diffusion model.<br>
                   {"Add the following tokens to your prompts for the model to work properly: <b>prefix</b>" if prefix else ""}
                  </p>
                  Running on {"<b>GPU 🔥</b>" if torch.cuda.is_available() else f"<b>CPU 🥶</b>. For faster inference it is recommended to <b>upgrade to GPU in <a href='https://huggingface.co/spaces/22h/vintedois-diffusion-v0-1'>Settings</a></b>"}<br><br>
                  <a style="display:inline-block" href="https://huggingface.co/spaces/22h/vintedois-diffusion-v0-1?duplicate=true"><img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
                </div>
            """
        )
        with gr.Row():
            with gr.Column(scale=55):
                with gr.Group():
                    with gr.Row():
                        prompt = gr.Textbox(label="Prompt", show_label=False, max_lines=2,
                                            placeholder=f"{prefix} [your prompt]").style(container=False)
                        generate = gr.Button(value="Generate").style(rounded=(False, True, True, False))

                    image_out = gr.Image(height=512)
                error_output = gr.Markdown()

            with gr.Column(scale=45):
                with gr.Tab("Options"):
                    with gr.Group():
                        neg_prompt = gr.Textbox(label="Negative prompt", placeholder="What to exclude from the image")
                        auto_prefix = gr.Checkbox(label="Prefix styling tokens automatically ()", value=prefix,
                                                  visible=prefix)

                        with gr.Row():
                            guidance = gr.Slider(label="Guidance scale", value=7.5, maximum=15)
                            steps = gr.Slider(label="Steps", value=50, minimum=2, maximum=75, step=1)

                        with gr.Row():
                            width = gr.Slider(label="Width", value=640, minimum=64, maximum=1024, step=8)
                            height = gr.Slider(label="Height", value=640, minimum=64, maximum=1024, step=8)

                        seed = gr.Slider(0, 2147483647, label='Seed (0 = random)', value=0, step=1)

                with gr.Tab("Image to image"):
                    with gr.Group():
                        image = gr.Image(label="Image", height=256, tool="editor", type="pil")
                        strength = gr.Slider(label="Transformation strength", minimum=0, maximum=1, step=0.01, value=0.5)

        auto_prefix.change(lambda x: gr.update(placeholder=f"{prefix} [your prompt]" if x else "[Your prompt]"),
                           inputs=auto_prefix, outputs=prompt, queue=False)

        inputs = [pipe, pipe_i2i, prompt, guidance, steps, width, height, seed, image, strength, neg_prompt, auto_prefix]
        outputs = [image_out, error_output]
        prompt.submit(inference, inputs=inputs, outputs=outputs)
        generate.click(inference, inputs=inputs, outputs=outputs)

        gr.HTML("""
        <div style="border-top: 1px solid #303030;">
          <br>
          <p>This space was created using <a href="https://huggingface.co/spaces/anzorq/sd-space-creator">SD Space Creator</a>.</p>
        </div>
        """)

    demo.queue(concurrency_count=1)
    demo.launch()


if __name__ == '__main__':
    args = parse_args()
    main(args)