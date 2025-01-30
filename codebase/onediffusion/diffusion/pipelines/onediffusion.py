import einops
import inspect
import torch
import numpy as np
import PIL
import os

from dataclasses import dataclass
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import (
    CONFIG_NAME,
    DEPRECATED_REVISION_ARGS,
    BaseOutput,
    PushToHubMixin,
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    is_torch_npu_available,
    is_torch_version,
    logging,
    numpy_to_pil,
    replace_example_docstring,
)
from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT, ModelMixin
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import BaseOutput
# from diffusers.image_processor import VaeImageProcessor
from transformers import T5EncoderModel, T5Tokenizer
from typing import Any, Callable, Dict, List, Optional, Union
from PIL import Image

from onediffusion.models.denoiser.nextdit import NextDiT
from onediffusion.dataset.utils import *
from onediffusion.dataset.multitask.multiview import calculate_rays
from onediffusion.diffusion.pipelines.image_processor import VaeImageProcessorOneDiffuser

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

SUPPORTED_DEVICE_MAP = ["balanced"]

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from one_diffusion import OneDiffusionPipeline

        >>> pipe = OneDiffusionPipeline.from_pretrained("path_to_one_diffuser_model")
        >>> pipe = pipe.to("cuda")

        >>> prompt = "A beautiful sunset over the ocean"
        >>> image = pipe(prompt).images[0]
        >>> image.save("beautiful_sunset.png")
        ```
"""

def create_c2w_matrix(azimuth_deg, elevation_deg, distance=1.0, target=np.array([0, 0, 0])):
    """
    Create a Camera-to-World (C2W) matrix from azimuth and elevation angles.

    Parameters:
    - azimuth_deg: Azimuth angle in degrees.
    - elevation_deg: Elevation angle in degrees.
    - distance: Distance from the target point.
    - target: The point the camera is looking at in world coordinates.

    Returns:
    - C2W: A 4x4 NumPy array representing the Camera-to-World transformation matrix.
    """
    # Convert angles from degrees to radians
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)

    # Spherical to Cartesian conversion for camera position
    x = distance * np.cos(elevation) * np.cos(azimuth)
    y = distance * np.cos(elevation) * np.sin(azimuth)
    z = distance * np.sin(elevation)
    camera_position = np.array([x, y, z])

    # Define the forward vector (from camera to target)
    target = 2*camera_position - target
    forward = target - camera_position
    forward /= np.linalg.norm(forward)

    # Define the world up vector
    world_up = np.array([0, 0, 1])

    # Compute the right vector
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 1e-6:
        # Handle the singularity when forward is parallel to world_up
        world_up = np.array([0, 1, 0])
        right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)

    # Recompute the orthogonal up vector
    up = np.cross(forward, right)

    # Construct the rotation matrix
    rotation = np.vstack([right, up, forward]).T  # 3x3

    # Construct the full C2W matrix
    C2W = np.eye(4)
    C2W[:3, :3] = rotation
    C2W[:3, 3] = camera_position

    return C2W

@dataclass
class OneDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[Image.Image], np.ndarray]
    latents: Optional[torch.Tensor] = None


def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")
    
    
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
    # max_clip: float = 1.5,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len) # 0.000169270833
    b = base_shift - m * base_seq_len # 0.5-0.0433333332
    mu = image_seq_len * m + b
    # mu = min(mu, max_clip)
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps



class OneDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using OneDiffuser.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        transformer ([`NextDiT`]):
            Conditional transformer (NextDiT) architecture to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. OneDiffuser uses the T5 model as text encoder.
        tokenizer (`T5Tokenizer`):
            Tokenizer of class T5Tokenizer.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    def __init__(
        self,
        transformer: NextDiT,
        vae: AutoencoderKL,
        text_encoder: T5EncoderModel,
        tokenizer: T5Tokenizer,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessorOneDiffuser(vae_scale_factor=self.vae_scale_factor)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.transformer, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.transformer, "_hf_hook"):
            return self.device
        for module in self.transformer.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        max_length=300,
    ):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask

        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {max_length} tokens: {removed_text}"
            )

        text_encoder_output = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask.to(device))
        prompt_embeds = text_encoder_output[0].to(torch.float32)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # duplicate attention mask for each generation per prompt
        attention_mask = attention_mask.repeat(1, num_images_per_prompt)
        attention_mask = attention_mask.view(bs_embed * num_images_per_prompt, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            uncond_encoder_output = self.text_encoder(uncond_input.input_ids.to(device), attention_mask=uncond_input.attention_mask.to(device))
            negative_prompt_embeds = uncond_encoder_output[0].to(torch.float32)

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # duplicate unconditional attention mask for each generation per prompt
            uncond_attention_mask = uncond_input.attention_mask.repeat(1, num_images_per_prompt)
            uncond_attention_mask = uncond_attention_mask.view(batch_size * num_images_per_prompt, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            attention_mask = torch.cat([uncond_attention_mask, attention_mask])

        return prompt_embeds.to(device), attention_mask.to(device)

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        forward_kwargs: Optional[Dict[str, Any]] = {},
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to self.transformer.config.sample_size):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.transformer.config.sample_size):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        height = height or self.transformer.config.input_size[-2] * 8 # TODO: Hardcoded downscale factor of vae
        width = width or self.transformer.config.input_size[-1] * 8

        # check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf
        do_classifier_free_guidance = guidance_scale > 1.0

        encoder_hidden_states, encoder_attention_mask = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )

        # set timesteps
        # # self.scheduler.set_timesteps(num_inference_steps, device=device)
        # timesteps = self.scheduler.timesteps
        timesteps = None

        # prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            self.dtype,
            device,
            generator,
            latents,
        )

        # prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[-1] * latents.shape[-2] / self.transformer.config.patch_size[-1] / self.transformer.config.patch_size[-2]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        
        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.transformer(
                    samples=latent_model_input.to(self.dtype),
                    timesteps=torch.tensor([t] * latent_model_input.shape[0], device=device),
                    encoder_hidden_states=encoder_hidden_states.to(self.dtype),
                    encoder_attention_mask=encoder_attention_mask,
                    **forward_kwargs
                )

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # scale and decode the image latents with vae
        latents = 1 / self.vae.config.scaling_factor * latents
        if latents.ndim == 5:
            latents = latents.squeeze(1)
        image = self.vae.decode(latents.to(self.vae.dtype)).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, None)

        return OneDiffusionPipelineOutput(images=image)

    @torch.no_grad()
    def img2img(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[PIL.Image.Image, List[PIL.Image.Image]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        denoise_mask: Optional[List[int]] = [1, 0],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        do_crop: bool = True,
        is_multiview: bool = False,
        multiview_azimuths: Optional[List[int]] = [0, 30, 60, 90],
        multiview_elevations: Optional[List[int]] = [0, 0, 0, 0],
        multiview_distances: float = 1.7,
        multiview_c2ws: Optional[List[torch.Tensor]] = None,
        multiview_intrinsics: Optional[torch.Tensor] = None,
        multiview_focal_length: float = 1.3887,
        forward_kwargs: Optional[Dict[str, Any]] = {},
        noise_scale: float = 1.0,
        **kwargs,
):
        # Convert single image to list for consistent handling
        if isinstance(image, PIL.Image.Image):
            image = [image]
            
        if height is None or width is None:
            closest_ar = get_closest_ratio(height=image[0].size[1], width=image[0].size[0], ratios=ASPECT_RATIO_512)
            height, width = int(closest_ar[0][0]), int(closest_ar[0][1])
        
        if not isinstance(multiview_distances, list) and not isinstance(multiview_distances, tuple):
            multiview_distances = [multiview_distances] * len(multiview_azimuths)
            
        # height = height or self.transformer.config.input_size[-2] * 8  # TODO: Hardcoded downscale factor of vae
        # width = width or self.transformer.config.input_size[-1] * 8

        # 1. check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Additional input validation for image list
        if not all(isinstance(img, PIL.Image.Image) for img in image):
            raise ValueError("All elements in image list must be PIL.Image objects")

        # 2. define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        encoder_hidden_states, encoder_attention_mask = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )

        # 4. Preprocess all images
        if image is not None and len(image) > 0:
            processed_image = self.image_processor.preprocess(image, height=height, width=width, do_crop=do_crop)
        else:
            processed_image = None
            
        # # Stack processed images along the sequence dimension
        # if len(processed_images) > 1:
        #     processed_image = torch.cat(processed_images, dim=0)
        # else:
        #     processed_image = processed_images[0]

        timesteps = None

        # 6. prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        if processed_image is not None:
            cond_latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                self.dtype,
                device,
                generator,
                latents,
                image=processed_image,
            )
        else:
            cond_latents = None
            
        # 7. prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        denoise_mask = torch.tensor(denoise_mask, device=device)
        denoise_indices = torch.where(denoise_mask == 1)[0]
        cond_indices = torch.where(denoise_mask == 0)[0]
        seq_length = denoise_mask.shape[0]

        latents = self.prepare_init_latents(
            batch_size * num_images_per_prompt,
            seq_length,
            num_channels_latents,
            height,
            width,
            self.dtype,
            device,
            generator,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        # image_seq_len = latents.shape[1] * latents.shape[-1] * latents.shape[-2] / self.transformer.config.patch_size[-1] / self.transformer.config.patch_size[-2]
        image_seq_len = noise_scale * sum(denoise_mask) * latents.shape[-1] * latents.shape[-2] / self.transformer.config.patch_size[-1] / self.transformer.config.patch_size[-2]
        # image_seq_len = 256
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        
        if is_multiview:
            cond_indices_images = [index // 2 for index in cond_indices if index % 2 == 0]
            cond_indices_rays = [index // 2 for index in cond_indices if index % 2 == 1]
                        
            multiview_elevations = [element for element in multiview_elevations if element is not None]
            multiview_azimuths = [element for element in multiview_azimuths if element is not None]
            multiview_distances = [element for element in multiview_distances if element is not None]
            
            if multiview_c2ws is None:
                multiview_c2ws = [
                    torch.tensor(create_c2w_matrix(azimuth, elevation, distance)) for azimuth, elevation, distance in zip(multiview_azimuths, multiview_elevations, multiview_distances)
                ]
                c2ws = torch.stack(multiview_c2ws).float()
            else:
                c2ws = torch.Tensor(multiview_c2ws).float()    
                
            c2ws[:, 0:3, 1:3] *= -1
            c2ws = c2ws[:, [1, 0, 2, 3], :]
            c2ws[:, 2, :] *= -1
            
            w2cs = torch.inverse(c2ws)
            if multiview_intrinsics is None:
                multiview_intrinsics = torch.Tensor([[[multiview_focal_length, 0, 0.5], [0, multiview_focal_length, 0.5], [0, 0, 1]]]).repeat(c2ws.shape[0], 1, 1)
            K = multiview_intrinsics
            Rs = w2cs[:, :3, :3]
            Ts = w2cs[:, :3, 3]
            sizes = torch.Tensor([[1, 1]]).repeat(c2ws.shape[0], 1)

            assert height == width
            cond_rays = calculate_rays(K, sizes, Rs, Ts, height // 8)
            cond_rays = cond_rays.reshape(-1, height // 8, width // 8, 6)
            # padding = (0, 10)
            # cond_rays = torch.nn.functional.pad(cond_rays, padding, "constant", 0)
            cond_rays = torch.cat([cond_rays, cond_rays, cond_rays[..., :4]], dim=-1) * 1.658
            cond_rays = cond_rays[None].repeat(batch_size * num_images_per_prompt, 1, 1, 1, 1)
            cond_rays = cond_rays.permute(0, 1, 4, 2, 3)
            cond_rays = cond_rays.to(device, dtype=self.dtype)

            latents = einops.rearrange(latents, "b (f n) c h w -> b f n c h w", n=2)
            if cond_latents is not None:
                latents[:, cond_indices_images, 0] = cond_latents
            latents[:, cond_indices_rays, 1] = cond_rays
            latents = einops.rearrange(latents, "b f n c h w -> b (f n) c h w")
        else:
            if cond_latents is not None:
                latents[:, cond_indices] = cond_latents
        
        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                input_t = torch.broadcast_to(einops.repeat(torch.Tensor([t]).to(device), "1 -> 1 f 1 1 1", f=latent_model_input.shape[1]), latent_model_input.shape).clone()
                
                if is_multiview:
                    input_t = einops.rearrange(input_t, "b (f n) c h w -> b f n c h w", n=2)
                    input_t[:, cond_indices_images, 0] = self.scheduler.timesteps[-1]
                    input_t[:, cond_indices_rays, 1] = self.scheduler.timesteps[-1]
                    input_t = einops.rearrange(input_t, "b f n c h w -> b (f n) c h w")
                else:
                    input_t[:, cond_indices] = self.scheduler.timesteps[-1]

                # predict the noise residual
                noise_pred = self.transformer(
                    samples=latent_model_input.to(self.dtype),
                    timesteps=input_t,
                    encoder_hidden_states=encoder_hidden_states.to(self.dtype),
                    encoder_attention_mask=encoder_attention_mask,
                    **forward_kwargs
                )

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                bs, n_frame = noise_pred.shape[:2]
                noise_pred = einops.rearrange(noise_pred, "b f c h w -> (b f) c h w")
                latents = einops.rearrange(latents, "b f c h w -> (b f) c h w")
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                latents = einops.rearrange(latents, "(b f) c h w -> b f c h w", b=bs, f=n_frame)
                if is_multiview:
                    latents = einops.rearrange(latents, "b (f n) c h w -> b f n c h w", n=2)
                    if cond_latents is not None:
                        latents[:, cond_indices_images, 0] = cond_latents
                    latents[:, cond_indices_rays, 1] = cond_rays
                    latents = einops.rearrange(latents, "b f n c h w -> b (f n) c h w")
                else:
                    if cond_latents is not None:
                        latents[:, cond_indices] = cond_latents

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        decoded_latents = latents / 1.658
        # scale and decode the image latents with vae
        latents = 1 / self.vae.config.scaling_factor * latents
        if latents.ndim == 5:
            latents = latents[:, denoise_indices]
            latents = einops.rearrange(latents, "b f c h w -> (b f) c h w")
        image = self.vae.decode(latents.to(self.vae.dtype)).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, None)

        return OneDiffusionPipelineOutput(images=image, latents=decoded_latents)
    
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None, image=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        if image is None:
            # scale the initial noise by the standard deviation required by the scheduler
            # latents = latents * self.scheduler.init_noise_sigma
            return latents
        
        image = image.to(device=device, dtype=dtype)
        
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        elif isinstance(generator, list):
            if image.shape[0] < batch_size and batch_size % image.shape[0] == 0:
                image = torch.cat([image] * (batch_size // image.shape[0]), dim=0)
            elif image.shape[0] < batch_size and batch_size % image.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image.shape[0]} to effective batch_size {batch_size} "
                )
            init_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = retrieve_latents(self.vae.encode(image.to(self.vae.dtype)), generator=generator)
            
        init_latents = self.vae.config.scaling_factor * init_latents
        init_latents = init_latents.to(device=device, dtype=dtype)

        init_latents = einops.rearrange(init_latents, "(bs views) c h w -> bs views c h w", bs=batch_size, views=init_latents.shape[0]//batch_size)
        # latents = einops.rearrange(latents, "b c h w -> b 1 c h w")
        # latents = torch.concat([latents, init_latents], dim=1)
        return init_latents
    
    def prepare_init_latents(self, batch_size, seq_length, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, seq_length, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        return latents

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]],
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        """
        Function for image generation using the OneDiffusionPipeline.
        """
        return self(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            height=height,
            width=width,
            eta=eta,
            generator=generator,
            latents=latents,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
        )

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model_path = pretrained_model_name_or_path
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        from_flax = kwargs.pop("from_flax", False)
        torch_dtype = kwargs.pop("torch_dtype", None)
        custom_pipeline = kwargs.pop("custom_pipeline", None)
        custom_revision = kwargs.pop("custom_revision", None)
        provider = kwargs.pop("provider", None)
        sess_options = kwargs.pop("sess_options", None)
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        use_onnx = kwargs.pop("use_onnx", None)
        load_connected_pipeline = kwargs.pop("load_connected_pipeline", False)
        
        if low_cpu_mem_usage and not is_accelerate_available():
            low_cpu_mem_usage = False
            logger.warning(
                "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                " install accelerate\n```\n."
            )

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        if device_map is not None and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Loading and dispatching requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `device_map=None`."
            )

        if device_map is not None and not is_accelerate_available():
            raise NotImplementedError(
                "Using `device_map` requires the `accelerate` library. Please install it using: `pip install accelerate`."
            )

        if device_map is not None and not isinstance(device_map, str):
            raise ValueError("`device_map` must be a string.")

        if device_map is not None and device_map not in SUPPORTED_DEVICE_MAP:
            raise NotImplementedError(
                f"{device_map} not supported. Supported strategies are: {', '.join(SUPPORTED_DEVICE_MAP)}"
            )

        if device_map is not None and device_map in SUPPORTED_DEVICE_MAP:
            if is_accelerate_version("<", "0.28.0"):
                raise NotImplementedError("Device placement requires `accelerate` version `0.28.0` or later.")

        if low_cpu_mem_usage is False and device_map is not None:
            raise ValueError(
                f"You cannot set `low_cpu_mem_usage` to False while using device_map={device_map} for loading and"
                " dispatching. Please make sure to set `low_cpu_mem_usage=True`."
            )

        transformer = NextDiT.from_pretrained(f"{model_path}", subfolder="transformer", torch_dtype=torch.float32, cache_dir=cache_dir)
        vae = AutoencoderKL.from_pretrained(f"{model_path}", subfolder="vae", cache_dir=cache_dir)
        text_encoder = T5EncoderModel.from_pretrained(f"{model_path}", subfolder="text_encoder", torch_dtype=torch.float16, cache_dir=cache_dir)
        tokenizer = T5Tokenizer.from_pretrained(model_path, subfolder="tokenizer", cache_dir=cache_dir)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler", cache_dir=cache_dir)

        pipeline = cls(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            **kwargs
        )

        return pipeline