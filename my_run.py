from diffusers import AutoPipelineForImage2Image
from diffusers import DDPMScheduler
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import retrieve_timesteps, retrieve_latents
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
import torch
from PIL import Image

num_steps_inversion = 5
strngth = 0.8
generator = None
device = "cuda" if torch.cuda.is_available() else "cpu"
image_path = "edit_dataset/01.jpg"
src_prompt = "butterfly perched on purple flower"
tgt_prompt = "dragonfly perched on purple flower"
ws1 = [1.5, 1.5, 1.5, 1.5]
ws2 = [1, 1, 1, 1]

def encode_image(image, pipe):
    image = pipe.image_processor.preprocess(image)
    image = image.to(device=device, dtype=pipeline.dtype)

    if pipe.vae.config.force_upcast:
        image = image.float()
        pipe.vae.to(dtype=torch.float32)

    if isinstance(generator, list):
        init_latents = [
            retrieve_latents(pipe.vae.encode(image[i : i + 1]), generator=generator[i])
            for i in range(1)
        ]
        init_latents = torch.cat(init_latents, dim=0)
    else:
        init_latents = retrieve_latents(pipe.vae.encode(image), generator=generator)

    if pipe.vae.config.force_upcast:
        pipe.vae.to(pipeline.dtype)

    init_latents = init_latents.to(pipeline.dtype)
    init_latents = pipe.vae.config.scaling_factor * init_latents

    return init_latents.to(dtype=torch.float16)

# def create_xts(scheduler, timesteps, x_0, noise_shift_delta=1, generator=None):
#     noising_delta = noise_shift_delta * (timesteps[0] - timesteps[1])
#     noise_timesteps = [timestep - int(noising_delta) for timestep in timesteps]
#     noise_timesteps = noise_timesteps[:3]

#     x_0_expanded = x_0.expand(len(noise_timesteps), -1, -1, -1)
#     noise = torch.randn(x_0_expanded.size(), generator=generator, device="cpu", dtype=x_0.dtype).to(x_0.device)
#     x_ts = scheduler.add_noise(x_0_expanded, noise, torch.IntTensor(noise_timesteps))
#     x_ts = [t.unsqueeze(dim=0) for t in list(x_ts)]
#     x_ts += [x_0]
#     return x_ts

def deterministic_ddpm_step(
    model_output: torch.FloatTensor,
    timestep,
    sample: torch.FloatTensor,
    eta,
    use_clipped_model_output,
    generator,
    variance_noise,
    return_dict,
    scheduler,
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned diffusion model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

    Returns:
        [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
            If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
            tuple is returned where the first element is the sample tensor.

    """
    t = timestep

    prev_t = scheduler.previous_timestep(t)

    if model_output.shape[1] == sample.shape[1] * 2 and scheduler.variance_type in [
        "learned",
        "learned_range",
    ]:
        model_output, predicted_variance = torch.split(
            model_output, sample.shape[1], dim=1
        )
    else:
        predicted_variance = None

    # 1. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[t]
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else scheduler.one
    )
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    # 2. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
    if scheduler.config.prediction_type == "epsilon":
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)
    elif scheduler.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif scheduler.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (
            beta_prod_t**0.5
        ) * model_output
    else:
        raise ValueError(
            f"prediction_type given as {scheduler.config.prediction_type} must be one of `epsilon`, `sample` or"
            " `v_prediction`  for the DDPMScheduler."
        )

    # 3. Clip or threshold "predicted x_0"
    if scheduler.config.thresholding:
        pred_original_sample = scheduler._threshold_sample(pred_original_sample)
    elif scheduler.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -scheduler.config.clip_sample_range, scheduler.config.clip_sample_range
        )

    # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_original_sample_coeff = (
        alpha_prod_t_prev ** (0.5) * current_beta_t
    ) / beta_prod_t
    current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

    # 5. Compute predicted previous sample µ_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_prev_sample = (
        pred_original_sample_coeff * pred_original_sample
        + current_sample_coeff * sample
    )

    return pred_prev_sample

def normalize(
    z_t,
    i,
    max_norm_zs,
):
    max_norm = max_norm_zs[i]
    if max_norm < 0:
        return z_t, 1

    norm = torch.norm(z_t)
    if norm < max_norm:
        return z_t, 1

    coeff = max_norm / norm
    z_t = z_t * coeff
    return z_t, coeff

def step_save_latents(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise= None,
        return_dict: bool = True,
    ):

    timestep_index = self._inner_index
    next_timestep_index = timestep_index + 1
    u_hat_t = deterministic_ddpm_step(
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        eta=eta,
        use_clipped_model_output=use_clipped_model_output,
        generator=generator,
        variance_noise=variance_noise,
        return_dict=False,
        scheduler=self,
    )
    x_t_minus_1 = self.x_ts[timestep_index]
    self.x_ts_c_hat.append(u_hat_t)
    
    z_t = x_t_minus_1 - u_hat_t
    self.latents.append(z_t)

    z_t, _ = normalize(z_t, timestep_index, [-1, -1, -1, 15.5])
    x_t_minus_1_predicted = u_hat_t + z_t

    if not return_dict:
        return (x_t_minus_1_predicted,)

    return DDIMSchedulerOutput(prev_sample=x_t_minus_1, pred_original_sample=None)

def step_use_latents(
    self,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    variance_noise= None,
    return_dict: bool = True,
):
    print(f'_inner_index: {self._inner_index}')
    timestep_index = self._inner_index
    next_timestep_index = timestep_index + 1
    z_t = self.latents[timestep_index]  # + 1 because latents[0] is X_T

    _, normalize_coefficient = normalize(
        z_t,
        timestep_index,
        [-1, -1, -1, 15.5],
    )

    if normalize_coefficient == 0:
        eta = 0

    # eta = normalize_coefficient

    x_t_hat_c_hat = deterministic_ddpm_step(
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        eta=eta,
        use_clipped_model_output=use_clipped_model_output,
        generator=generator,
        variance_noise=variance_noise,
        return_dict=False,
        scheduler=self,
    )

    w1 = ws1[timestep_index]
    w2 = ws2[timestep_index]

    x_t_minus_1_exact = self.x_ts[timestep_index]
    x_t_minus_1_exact = x_t_minus_1_exact.expand_as(x_t_hat_c_hat)

    x_t_c_hat: torch.Tensor = self.x_ts_c_hat[timestep_index]

    x_t_c = x_t_c_hat[0].expand_as(x_t_hat_c_hat)
    
    zero_index_reconstruction = 0
    edit_prompts_num = (model_output.size(0) - zero_index_reconstruction) // 2
    x_t_hat_c_indices = (zero_index_reconstruction, edit_prompts_num + zero_index_reconstruction)
    edit_images_indices = (
        edit_prompts_num + zero_index_reconstruction,
        model_output.size(0)
    )
    x_t_hat_c = torch.zeros_like(x_t_hat_c_hat)
    x_t_hat_c[edit_images_indices[0] : edit_images_indices[1]] = x_t_hat_c_hat[
        x_t_hat_c_indices[0] : x_t_hat_c_indices[1]
    ]
    v1 = x_t_hat_c_hat - x_t_hat_c
    v2 = x_t_hat_c - normalize_coefficient * x_t_c

    x_t_minus_1 = normalize_coefficient * x_t_minus_1_exact + w1 * v1 + w2 * v2

    x_t_minus_1[x_t_hat_c_indices[0] : x_t_hat_c_indices[1]] = x_t_minus_1[
        edit_images_indices[0] : edit_images_indices[1]
    ] # update x_t_hat_c to be x_t_hat_c_hat
    

    if not return_dict:
        return (x_t_minus_1,)

    return DDIMSchedulerOutput(
        prev_sample=x_t_minus_1,
        pred_original_sample=None,
    )


class myDDPMScheduler(DDPMScheduler):
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise= None,
        return_dict: bool = True,
        ):
        print(f"timestep: {timestep}")

        res_inv =  step_save_latents(
                self,
                model_output[:1, :, :, :],
                timestep,
                sample[:1, :, :, :],
                eta,
                use_clipped_model_output,
                generator,
                variance_noise,
                return_dict,
            )

        res_inf = step_use_latents(
                self,
                model_output[1:, :, :, :],
                timestep,
                sample[1:, :, :, :],
                eta,
                use_clipped_model_output,
                generator,
                variance_noise,
                return_dict,
            )
        
        self._inner_index+=1

        res = (torch.cat((res_inv[0], res_inf[0]), dim=0),)
        return res


pipeline = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", safety_checker = None)
pipeline = pipeline.to(device)
pipeline.scheduler = DDPMScheduler.from_pretrained(  # type: ignore
                'stabilityai/sdxl-turbo',
                subfolder="scheduler",
                # cache_dir="/home/joberant/NLP_2223/giladd/test_dir/sdxl-turbo/models_cache",
            )
# pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

denoising_start = 0.2
timesteps, num_inference_steps = retrieve_timesteps(
                        pipeline.scheduler, num_steps_inversion, device, None
                    )
timesteps, num_inference_steps = pipeline.get_timesteps(
                        num_inference_steps=num_inference_steps,
                        device=device,
                        denoising_start=denoising_start,
                        strength=0,
                    )
timesteps = timesteps.type(torch.int64)
from functools import partial

timesteps = [torch.tensor(t) for t in timesteps.tolist()]
pipeline.__call__ = partial(
    pipeline.__call__,
    num_inference_steps=num_steps_inversion,
    guidance_scale=0,
    generator=generator,
    denoising_start=denoising_start,
    strength=0,
)

# timesteps, num_inference_steps = retrieve_timesteps(pipeline.scheduler, num_steps_inversion, device, None)
# timesteps, num_inference_steps = pipeline.get_timesteps(num_inference_steps=num_inference_steps, device=device, strength=strngth)


from utils import get_ddpm_inversion_scheduler, create_xts



from config import get_config, get_config_name
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--images_paths", type=str, default=None)
# parser.add_argument("--images_folder", type=str, default=None)
# parser.set_defaults(force_use_cpu=False)
# parser.add_argument("--force_use_cpu", action="store_true")
# parser.add_argument("--folder_name", type=str, default='test_measure_time')
# parser.add_argument("--config_from_file", type=str, default='run_configs/noise_shift_guidance_1_5.yaml')
# parser.set_defaults(save_intermediate_results=False)
# parser.add_argument("--save_intermediate_results", action="store_true")
# parser.add_argument("--batch_size", type=int, default=None)
# parser.set_defaults(skip_p_to_p=False)
# parser.add_argument("--skip_p_to_p", action="store_true", default=True)
# parser.set_defaults(only_p_to_p=False)
# parser.add_argument("--only_p_to_p", action="store_true")
# parser.set_defaults(fp16=False)
# parser.add_argument("--fp16", action="store_true", default=False)
# parser.add_argument("--prompts_file", type=str, default='dataset_measure_time/dataset.json')
# parser.add_argument("--images_in_prompts_file", type=str, default=None)
# parser.add_argument("--seed", type=int, default=2)
# parser.add_argument("--time_measure_n", type=int, default=1)

# args = parser.parse_args()
class Object(object):
    pass

args = Object()
args.images_paths = None
args.images_folder = None
args.force_use_cpu = False
args.folder_name = 'test_measure_time'
args.config_from_file = 'run_configs/noise_shift_guidance_1_5.yaml'
args.save_intermediate_results = False
args.batch_size = None
args.skip_p_to_p = True
args.only_p_to_p = False
args.fp16 = False
args.prompts_file = 'dataset_measure_time/dataset.json'
args.images_in_prompts_file = None
args.seed = 986
args.time_measure_n = 1


assert (
    args.batch_size is None or args.save_intermediate_results is False
), "save_intermediate_results is not implemented for batch_size > 1"

config = get_config(args)





# latent = latents[0].expand(3, -1, -1, -1)
# prompt = [src_prompt, src_prompt, tgt_prompt]

# image = pipeline.__call__(image=latent, prompt=prompt, eta=1).images

# for i, im in enumerate(image):
#     im.save(f"output_{i}.png")

def run(image_path, src_prompt, tgt_prompt, seed, w1, w2):
    generator = torch.Generator().manual_seed(seed)
    x_0_image = Image.open(image_path).convert("RGB").resize((512, 512), Image.LANCZOS)
    x_0 = encode_image(x_0_image, pipeline)
    # x_ts = create_xts(pipeline.scheduler, timesteps, x_0, noise_shift_delta=1, generator=generator)
    x_ts = create_xts(1, None, 0, generator, pipeline.scheduler, timesteps, x_0, no_add_noise=False)
    x_ts = [xt.to(dtype=torch.float16) for xt in x_ts]
    latents = [x_ts[0]]
    x_ts_c_hat = [None]
    config.ws1 = [w1] * 4
    config.ws2 = [w2] * 4
    pipeline.scheduler = get_ddpm_inversion_scheduler(
                    pipeline.scheduler,
                    config.step_function,
                    config,
                    timesteps,
                    config.save_timesteps,
                    latents,
                    x_ts,
                    x_ts_c_hat,
                    args.save_intermediate_results,
                    pipeline,
                    x_0,
                    v1s_images := [],
                    v2s_images := [],
                    deltas_images := [],
                    v1_x0s := [],
                    v2_x0s := [],
                    deltas_x0s := [],
                    "res12",
                    image_name="im_name",
                    time_measure_n=args.time_measure_n,
                )
    latent = latents[0].expand(3, -1, -1, -1)
    prompt = [src_prompt, src_prompt, tgt_prompt]
    image = pipeline.__call__(image=latent, prompt=prompt, eta=1).images
    return image[2]

if __name__ == "__main__":
    res = run(image_path, src_prompt, tgt_prompt, args.seed, 1.5, 1.0)
    res.save("output.png")
