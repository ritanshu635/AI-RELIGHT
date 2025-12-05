import os
import math
import gradio as gr
import numpy as np
import torch
import safetensors.torch as sf
import db_examples

# Load environment variables from .env file
from pathlib import Path
env_file = Path(__file__).parent / '.env'
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
    print(f"Loaded environment variables from {env_file}")

# Fix SSL certificate issue - use certifi package
import ssl
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['CURL_CA_BUNDLE'] = certifi.where()

from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from briarmbg import BriaRMBG
from enum import Enum
from torch.hub import download_url_to_file
from gpt_recommendations import GPTRecommendationClient
from upscaler import ImageUpscaler


# 'stablediffusionapi/realistic-vision-v51'
# 'runwayml/stable-diffusion-v1-5'
sd15_name = 'stablediffusionapi/realistic-vision-v51'
tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

# Change UNet

with torch.no_grad():
    new_conv_in = torch.nn.Conv2d(12, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
    new_conv_in.weight.zero_()
    new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    new_conv_in.bias = unet.conv_in.bias
    unet.conv_in = new_conv_in

unet_original_forward = unet.forward


def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
    c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
    new_sample = torch.cat([sample, c_concat], dim=1)
    kwargs['cross_attention_kwargs'] = {}
    return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)


unet.forward = hooked_unet_forward

# Load

model_path = './models/iclight_sd15_fbc.safetensors'

if not os.path.exists(model_path):
    download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors', dst=model_path)

sd_offset = sf.load_file(model_path)
sd_origin = unet.state_dict()
keys = sd_origin.keys()
sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged, keys

# Device

device = torch.device('cuda')
text_encoder = text_encoder.to(device=device, dtype=torch.float16)
vae = vae.to(device=device, dtype=torch.bfloat16)
unet = unet.to(device=device, dtype=torch.float16)
rmbg = rmbg.to(device=device, dtype=torch.float32)

# Initialize GPT client for AI recommendations
gpt_client = GPTRecommendationClient()

# Initialize upscaler for 1080p output
upscaler = ImageUpscaler(model_path='./models/RealESRGAN_x4plus.pth', device='cuda')

# SDP

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Samplers

ddim_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

euler_a_scheduler = EulerAncestralDiscreteScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    steps_offset=1
)

dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
    steps_offset=1
)

# Pipelines

t2i_pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)

i2i_pipe = StableDiffusionImg2ImgPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)


@torch.inference_mode()
def encode_prompt_inner(txt: str):
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]

    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids).last_hidden_state

    return conds


@torch.inference_mode()
def encode_prompt_pair(positive_prompt, negative_prompt):
    c = encode_prompt_inner(positive_prompt)
    uc = encode_prompt_inner(negative_prompt)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return c, uc


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h


def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


@torch.inference_mode()
def run_rmbg(img, sigma=0.0):
    H, W, C = img.shape
    assert C == 3
    k = (256.0 / float(H * W)) ** 0.5
    feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
    feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
    alpha = rmbg(feed)[0][0]
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)[0]
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
    result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
    return result.clip(0, 255).astype(np.uint8), alpha


@torch.inference_mode()
def process(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source):
    bg_source = BGSource(bg_source)

    if bg_source == BGSource.UPLOAD:
        pass
    elif bg_source == BGSource.UPLOAD_FLIP:
        input_bg = np.fliplr(input_bg)
    elif bg_source == BGSource.GREY:
        input_bg = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8) + 64
    elif bg_source == BGSource.LEFT:
        gradient = np.linspace(224, 32, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.RIGHT:
        gradient = np.linspace(32, 224, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.TOP:
        gradient = np.linspace(224, 32, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.BOTTOM:
        gradient = np.linspace(32, 224, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    else:
        raise 'Wrong background source!'

    rng = torch.Generator(device=device).manual_seed(seed)

    fg = resize_and_center_crop(input_fg, image_width, image_height)
    bg = resize_and_center_crop(input_bg, image_width, image_height)
    concat_conds = numpy2pytorch([fg, bg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

    conds, unconds = encode_prompt_pair(positive_prompt=prompt + ', ' + a_prompt, negative_prompt=n_prompt)

    latents = t2i_pipe(
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=steps,
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)
    pixels = [resize_without_crop(
        image=p,
        target_width=int(round(image_width * highres_scale / 64.0) * 64),
        target_height=int(round(image_height * highres_scale / 64.0) * 64))
    for p in pixels]

    pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.to(device=unet.device, dtype=unet.dtype)

    image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8
    fg = resize_and_center_crop(input_fg, image_width, image_height)
    bg = resize_and_center_crop(input_bg, image_width, image_height)
    concat_conds = numpy2pytorch([fg, bg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

    latents = i2i_pipe(
        image=latents,
        strength=highres_denoise,
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=int(round(steps / highres_denoise)),
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels, quant=False)

    return pixels, [fg, bg]


@torch.inference_mode()
def process_relight(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source, progress=gr.Progress()):
    try:
        print("=" * 50)
        print("Starting background-conditioned relight process...")
        progress(0, desc="Starting...")
        
        # Validate inputs
        if input_fg is None:
            print("ERROR: No foreground image provided")
            return []
        
        if input_bg is None:
            print("WARNING: No background image provided, using default")
            # Create a default grey background
            input_bg = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8) + 64
        
        print(f"Foreground shape: {input_fg.shape}")
        print(f"Background shape: {input_bg.shape}")
        
        # Remove background from foreground
        print("Removing background from foreground...")
        progress(0.1, desc="Removing background...")
        input_fg, matting = run_rmbg(input_fg)
        
        # Process with background
        print("Processing with background...")
        progress(0.3, desc="Blending with background (this may take 10-20 seconds)...")
        results, extra_images = process(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source)
        results = [(x * 255.0).clip(0, 255).astype(np.uint8) for x in results]
        
        print(f"Process returned {len(results)} images")
        
        progress(0.7, desc="Upscaling to high resolution...")
        
        # Apply upscaling and resize for browser compatibility
        upscaled_results = []
        for i, img in enumerate(results):
            try:
                print(f"Upscaling image {i+1}/{len(results)}")
                progress(0.7 + (0.2 * (i / len(results))), desc=f"Upscaling image {i+1}/{len(results)}...")
                upscaled_img = upscaler.upscale_to_1080p(img)
                
                # Resize for browser display - max 1280p for Gradio 3.41.2 compatibility
                h, w = upscaled_img.shape[:2]
                max_height = 1280
                if h > max_height:
                    print(f"Image too large ({h}p), resizing to {max_height}p for browser display")
                    scale = max_height / h
                    new_h = max_height
                    new_w = int(w * scale)
                    pil_img = Image.fromarray(upscaled_img)
                    pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                    upscaled_img = np.array(pil_img)
                
                # Ensure image is in correct format
                if upscaled_img.dtype != np.uint8:
                    upscaled_img = upscaled_img.astype(np.uint8)
                
                upscaled_results.append(upscaled_img)
            except Exception as e:
                print(f"Upscaling error: {e}, using original image")
                import traceback
                traceback.print_exc()
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                upscaled_results.append(img)
        
        progress(0.9, desc="Saving images...")
        
        # Save images
        from datetime import datetime
        output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, img in enumerate(upscaled_results):
            output_path = os.path.join(output_dir, f"bg_relit_{timestamp}_{i+1}.png")
            Image.fromarray(img).save(output_path)
            print(f"Saved image to: {output_path}")
        
        # Save to temp files and return paths for better Gradio compatibility
        saved_image_paths = []
        for i, img in enumerate(upscaled_results):
            temp_path = os.path.join(output_dir, f"temp_display_{timestamp}_{i+1}.png")
            Image.fromarray(img).save(temp_path)
            saved_image_paths.append(temp_path)
        
        # Also save extra images
        for i, img in enumerate(extra_images):
            temp_path = os.path.join(output_dir, f"temp_extra_{timestamp}_{i+1}.png")
            Image.fromarray(img).save(temp_path)
            saved_image_paths.append(temp_path)
        
        progress(1.0, desc="Complete!")
        
        print(f"Returning {len(saved_image_paths)} result image paths")
        print("=" * 50)
        
        return saved_image_paths
        
    except Exception as e:
        print(f"ERROR in process_relight: {e}")
        import traceback
        traceback.print_exc()
        return []


@torch.inference_mode()
def process_normal(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source, progress=gr.Progress()):
    try:
        print("=" * 50)
        print("Computing normal map (4x slower)...")
        progress(0, desc="Starting normal map computation...")
        
        if input_fg is None:
            print("ERROR: No foreground image provided")
            return []
        
        if input_bg is None:
            print("WARNING: No background image provided, using default")
            input_bg = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8) + 64
        
        progress(0.05, desc="Removing background...")
        input_fg, matting = run_rmbg(input_fg, sigma=16)

        print('Processing left lighting...')
        progress(0.1, desc="Processing left lighting (1/4)...")
        left = process(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.LEFT.value)[0][0]

        print('Processing right lighting...')
        progress(0.3, desc="Processing right lighting (2/4)...")
        right = process(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.RIGHT.value)[0][0]

        print('Processing bottom lighting...')
        progress(0.5, desc="Processing bottom lighting (3/4)...")
        bottom = process(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.BOTTOM.value)[0][0]

        print('Processing top lighting...')
        progress(0.7, desc="Processing top lighting (4/4)...")
        top = process(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.TOP.value)[0][0]

        inner_results = [left * 2.0 - 1.0, right * 2.0 - 1.0, bottom * 2.0 - 1.0, top * 2.0 - 1.0]

        ambient = (left + right + bottom + top) / 4.0
        h, w, _ = ambient.shape
        matting = resize_and_center_crop((matting[..., 0] * 255.0).clip(0, 255).astype(np.uint8), w, h).astype(np.float32)[..., None] / 255.0

        def safa_divide(a, b):
            e = 1e-5
            return ((a + e) / (b + e)) - 1.0

        left = safa_divide(left, ambient)
        right = safa_divide(right, ambient)
        bottom = safa_divide(bottom, ambient)
        top = safa_divide(top, ambient)

        u = (right - left) * 0.5
        v = (top - bottom) * 0.5

        sigma = 10.0
        u = np.mean(u, axis=2)
        v = np.mean(v, axis=2)
        h = (1.0 - u ** 2.0 - v ** 2.0).clip(0, 1e5) ** (0.5 * sigma)
        z = np.zeros_like(h)

        normal = np.stack([u, v, h], axis=2)
        normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
        normal = normal * matting + np.stack([z, z, 1 - z], axis=2) * (1 - matting)

        results = [normal, left, right, bottom, top] + inner_results
        results = [(x * 127.5 + 127.5).clip(0, 255).astype(np.uint8) for x in results]
        
        print(f"Generated {len(results)} normal map images")
        
        # Apply upscaling and resize for browser compatibility
        upscaled_results = []
        for i, img in enumerate(results):
            try:
                print(f"Upscaling normal map image {i+1}/{len(results)}")
                upscaled_img = upscaler.upscale_to_1080p(img)
                
                # Resize for browser display - max 1280p for Gradio 3.41.2 compatibility
                h, w = upscaled_img.shape[:2]
                max_height = 1280
                if h > max_height:
                    scale = max_height / h
                    new_h = max_height
                    new_w = int(w * scale)
                    pil_img = Image.fromarray(upscaled_img)
                    pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                    upscaled_img = np.array(pil_img)
                
                # Ensure image is in correct format
                if upscaled_img.dtype != np.uint8:
                    upscaled_img = upscaled_img.astype(np.uint8)
                
                upscaled_results.append(upscaled_img)
            except Exception as e:
                print(f"Upscaling error: {e}, using original image")
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                upscaled_results.append(img)
        
        # Save images
        from datetime import datetime
        output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, img in enumerate(upscaled_results):
            output_path = os.path.join(output_dir, f"normal_{timestamp}_{i+1}.png")
            Image.fromarray(img).save(output_path)
            print(f"Saved normal map to: {output_path}")
        
        # Save to temp files and return paths
        saved_image_paths = []
        for i, img in enumerate(upscaled_results):
            temp_path = os.path.join(output_dir, f"temp_normal_{timestamp}_{i+1}.png")
            Image.fromarray(img).save(temp_path)
            saved_image_paths.append(temp_path)
        
        print("Normal map computation complete!")
        print(f"Returning {len(saved_image_paths)} normal map image paths")
        print("=" * 50)
        
        return saved_image_paths
        
    except Exception as e:
        print(f"ERROR in process_normal: {e}")
        import traceback
        traceback.print_exc()
        return []


quick_prompts = [
    'beautiful woman',
    'handsome man',
    'beautiful woman, cinematic lighting',
    'handsome man, cinematic lighting',
    'beautiful woman, natural lighting',
    'handsome man, natural lighting',
    'beautiful woman, neo punk lighting, cyberpunk',
    'handsome man, neo punk lighting, cyberpunk',
]
quick_prompts = [[x] for x in quick_prompts]


# AI Recommendations workflow functions
def get_ai_recommendations(image):
    """Get AI lighting recommendations for the uploaded image"""
    if image is None:
        print("No image provided for AI recommendations")
        return gr.update(visible=False), gr.update(visible=False)
    
    try:
        print("Getting AI recommendations...")
        suggestions = gpt_client.get_lighting_recommendations(image)
        print(f"AI suggestions: {suggestions}")
        return gr.update(choices=suggestions, visible=True, value=suggestions[0]), gr.update(visible=True)
    except ValueError as e:
        # API key error - show error message
        print(f"API key error: {e}")
        error_msg = ["API key invalid", "Check .env file", "Using defaults"]
        return gr.update(choices=error_msg, visible=True, value=error_msg[0]), gr.update(visible=True)
    except Exception as e:
        print(f"Error getting AI recommendations: {e}")
        import traceback
        traceback.print_exc()
        # Return default suggestions on error
        default_suggestions = ["warm golden light", "dramatic side lighting", "soft studio light"]
        return gr.update(choices=default_suggestions, visible=True, value=default_suggestions[0]), gr.update(visible=True)


def apply_ai_suggestion(selected_suggestion, current_prompt):
    """Apply the selected AI suggestion to the prompt field"""
    if selected_suggestion:
        # If there's already a prompt, append the suggestion; otherwise, use it as the prompt
        if current_prompt and current_prompt.strip():
            # Check if the suggestion is already in the prompt to avoid duplication
            if selected_suggestion not in current_prompt:
                return current_prompt + ", " + selected_suggestion
            return current_prompt
        else:
            return selected_suggestion
    return current_prompt


def show_background_upload():
    """Show background upload and blend button when Add Background Image is clicked"""
    return gr.update(visible=True), gr.update(visible=True)


class BGSource(Enum):
    UPLOAD = "Use Background Image"
    UPLOAD_FLIP = "Use Flipped Background Image"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"
    GREY = "Ambient"


# Custom CSS for modern UI
custom_css = """
/* Dark theme base */
.gradio-container {
    background: linear-gradient(135deg, #0B0F19 0%, #1a1f2e 100%) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Typography */
h1, h2, h3 {
    color: #E6E9EF !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}

label {
    color: #B8BCC8 !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
}

/* Buttons */
.gr-button {
    background: linear-gradient(135deg, #6C5CE7 0%, #5B4BC4 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(108, 92, 231, 0.3) !important;
}

.gr-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(108, 92, 231, 0.4) !important;
}

/* Input fields and sliders */
.gr-input, .gr-slider {
    background-color: #1E2330 !important;
    border: 1px solid #2D3548 !important;
    color: #E6E9EF !important;
    border-radius: 6px !important;
}

/* Image containers */
.gr-image {
    border-radius: 12px !important;
    border: 2px solid #2D3548 !important;
    overflow: hidden !important;
}

/* Gallery */
.gr-gallery {
    background-color: #1E2330 !important;
    border-radius: 12px !important;
    border: 1px solid #2D3548 !important;
}

/* Accordion */
.gr-accordion {
    background-color: #1E2330 !important;
    border: 1px solid #2D3548 !important;
    border-radius: 8px !important;
}

/* Special button for AI Recommendations */
.ai-recommend-btn {
    background: linear-gradient(135deg, #00D9FF 0%, #0099CC 100%) !important;
    box-shadow: 0 4px 12px rgba(0, 217, 255, 0.3) !important;
}

.ai-recommend-btn:hover {
    box-shadow: 0 6px 20px rgba(0, 217, 255, 0.4) !important;
}
"""

# Custom JavaScript to remove fetch timeout
custom_js = """
<script>
// Override fetch to remove timeout
(function() {
    const originalFetch = window.fetch;
    window.fetch = function(...args) {
        // Remove any timeout from fetch options
        if (args[1] && args[1].signal) {
            delete args[1].signal;
        }
        return originalFetch.apply(this, args);
    };
    
    console.log('Fetch timeout override applied - no timeout for API calls');
})();
</script>
"""

block = gr.Blocks(css=custom_css, head=custom_js)
with block:
    with gr.Row():
        gr.Markdown("## IC-Light (Relighting with Foreground and Background Condition)")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                input_fg = gr.Image(source='upload', type="numpy", label="Foreground", height=480)
                input_bg = gr.Image(source='upload', type="numpy", label="Background", height=480, visible=False)
            
            # Add Background Image button
            add_bg_button = gr.Button(value="Add Background Image", elem_classes=["gr-button"])
            
            # Blend button (initially hidden)
            blend_button = gr.Button(value="Blend with Background", visible=False)
            
            # AI Recommendations UI components
            ai_recommend_button = gr.Button(value="✨ Get AI Recommendations", elem_classes=["ai-recommend-btn"])
            ai_suggestions = gr.Radio(label="AI Lighting Suggestions", choices=[], visible=False)
            apply_ai_button = gr.Button(value="Apply Selected Suggestion", visible=False)
            
            prompt = gr.Textbox(label="Prompt")
            bg_source = gr.Radio(choices=[e.value for e in BGSource],
                                 value=BGSource.UPLOAD.value,
                                 label="Background Source", type='value')

            example_prompts = gr.Dataset(samples=quick_prompts, label='Prompt Quick List', components=[prompt])
            bg_gallery = gr.Gallery(height=450, object_fit='contain', label='Background Quick List', value=db_examples.bg_samples, columns=5, allow_preview=False)
            relight_button = gr.Button(value="Relight")

            with gr.Group():
                with gr.Row():
                    num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                    seed = gr.Number(label="Seed", value=12345, precision=0)
                with gr.Row():
                    image_width = gr.Slider(label="Image Width", minimum=256, maximum=1024, value=512, step=64)
                    image_height = gr.Slider(label="Image Height", minimum=256, maximum=1024, value=640, step=64)

            with gr.Accordion("Advanced options", open=False):
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=7.0, step=0.01)
                highres_scale = gr.Slider(label="Highres Scale", minimum=1.0, maximum=3.0, value=1.5, step=0.01)
                highres_denoise = gr.Slider(label="Highres Denoise", minimum=0.1, maximum=0.9, value=0.5, step=0.01)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='lowres, bad anatomy, bad hands, cropped, worst quality')
                normal_button = gr.Button(value="Compute Normal (4x Slower)")
        with gr.Column():
            result_gallery = gr.Gallery(height=832, object_fit='contain', label='Outputs', type='filepath')
    with gr.Row():
        dummy_image_for_outputs = gr.Image(visible=False, label='Result')
        gr.Examples(
            fn=lambda *args: [args[-1]],
            examples=db_examples.background_conditioned_examples,
            inputs=[
                input_fg, input_bg, prompt, bg_source, image_width, image_height, seed, dummy_image_for_outputs
            ],
            outputs=[result_gallery],
            run_on_click=True, examples_per_page=1024
        )
    ips = [input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source]
    
    # Add API calls with no timeout and progress
    relight_button.click(
        fn=process_relight, 
        inputs=ips, 
        outputs=[result_gallery],
        api_name="relight_bg",
        show_progress=True
    )
    normal_button.click(
        fn=process_normal, 
        inputs=ips, 
        outputs=[result_gallery],
        api_name="normal",
        show_progress=True
    )
    blend_button.click(
        fn=process_relight, 
        inputs=ips, 
        outputs=[result_gallery],
        api_name="blend",
        show_progress=True
    )
    example_prompts.click(lambda x: x[0], inputs=example_prompts, outputs=prompt, show_progress=False, queue=False)

    def bg_gallery_selected(gal, evt: gr.SelectData):
        return gal[evt.index]['name']

    bg_gallery.select(bg_gallery_selected, inputs=bg_gallery, outputs=input_bg)
    
    # Wire AI recommendations workflow
    ai_recommend_button.click(
        fn=get_ai_recommendations,
        inputs=[input_fg],
        outputs=[ai_suggestions, apply_ai_button]
    )
    apply_ai_button.click(
        fn=apply_ai_suggestion,
        inputs=[ai_suggestions, prompt],
        outputs=[prompt]
    )
    
    # Wire background upload show/hide logic
    add_bg_button.click(
        fn=show_background_upload,
        outputs=[input_bg, blend_button]
    )


# Configure queue with NO timeout for long processing
block.queue(
    concurrency_count=4,
    max_size=10,
    api_open=True
)

# Launch with extended settings
block.launch(
    server_name='0.0.0.0',
    show_error=True,
    max_threads=10,
    inbrowser=False,
    quiet=False
)
