## Supported Models

### Judges (TRIGScore)
Default judge: `Qwen/Qwen3.5-9B` served with vLLM (`scripts/slurm/serve_judge.sh`). Thinking is disabled per request. Any OpenAI-compatible VLM endpoint works through `trig_api`.

### General Models
1. [OneDiffusion](https://arxiv.org/abs/2411.16318)  
Alias: onediffusion  
Class: general_models.OneDiffusionModel
2. [OmnigGen](https://arxiv.org/abs/2409.11340)  
Alias: omnigen  
Class: general_models.OmniGenModel

### Text-to-Image Model
1.  [DALL·E 3](https://openai.com/dall-e-3/)  
Alias: dalle3  
Class: text_to_image_models.DALLE3Model
2. [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)  
Alias: sd15  
Class: text_to_image_models.SD15Model
2. [Stable Diffusion XL](https://arxiv.org/abs/2307.01952)  
Alias: sdxl  
Class: text_to_image_models.SDXLModel
3. [Stable Diffusion 3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)  
Alias: sd35   
Class: text_to_image_models.SD35Model
4. [PixArt-Σ](https://arxiv.org/abs/2403.04692)  
Alias: pixart_sigma  
Class: text_to_image_models.PixartSigmaModel
5. [Sana](https://arxiv.org/abs/2410.10629)  
Alias: sana  
Class: text_to_image_models.SanaModel
6. [FLUX](https://bfl.ai/)  
Alias: flux  
Class: text_to_image_models.FLUXModel
7. [Janus-Pro](https://arxiv.org/abs/2501.17811)  
Alias: janus  
Class: text_to_image_models.JanusProModel
8. [Janus-Flow](https://arxiv.org/abs/2411.07975)  
Alias: janus_flow  
Class: text_to_image_models.JanusFlowModel
9. [Qwen-Image](https://arxiv.org/abs/2508.02324)  
Alias: qwen_image  
Class: text_to_image_models.QwenImageModel
### 2026 refresh (diffusers main, `venv/t2i`)
| Alias | Model | Notes |
|---|---|---|
| `qwen_image_21` | [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) | 7B unified gen+edit, 40 steps, 2048×2048 native |
| `qwen_image_21_turbo` | [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) | 8-step distilled, CFG=1 |
| `flux2_dev` | [FLUX.2 [dev]](https://huggingface.co/black-forest-labs/FLUX.2-dev) | 32B, gated, 50 steps, cfg 4 |
| `flux2_klein_base_9b` | [FLUX.2 klein base 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) | undistilled, CFG, fine-tunable |
| `flux2_klein_9b` | [FLUX.2 klein 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | 4-step distilled |
| `zimage_base` | [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) | 6B undistilled, Apache-2.0, 50 steps, cfg 4 |

All of these load by repo id from `HF_HOME=/dpc/kuin0164/zsc/hf_home`; no local path config needed.
### TRIG Models
1. SD1.5 models after RL by DDPO  
Alias: sd15_ddpo  
Class: text_to_image_models.SD15DDPOModel
2. FLUX models after LoRA Finetuning on Knowlege and Ambiguity Dimension  
Alias: flux_ft  
Class: text_to_image_models.FLUXFTModel
### Image-editing Models
1. [InstructPix2Pix](https://arxiv.org/abs/2211.09800)  
Alias: instructp2p  
Class: image_editing_models.InstructPix2PixModel
2. [FreeDiff](https://arxiv.org/abs/2404.11895)  
Alias: freediff  
Class: image_editing_models.FreeDiffModel
3. [FlowEdit](https://arxiv.org/abs/2412.08629)  
Alias: flowedit  
Class: image_editing_models.FlowEditModel
4. [HQ-Edit](https://arxiv.org/abs/2404.09990)  
Alias: hqedit   
Class: image_editing_models.HQEditModel
5. [RF-Inversion](https://arxiv.org/abs/2410.10792)  
Alias: rfinversion   
Class: image_editing_models.RFInversionModel
6. [RF-Solver](https://arxiv.org/abs/2411.04746)  
Alias: rfsolver  
Class: image_editing_models.RFSolverModel
### Subjects Driven Models
'blipdiffusion': 'subject_driven_models.BlipDiffusionModel'
'ssrencoder': 'subject_driven_models.SSREncoderModel'
'ominicontrol': 'subject_driven_models.OminiControlModel'
'xflux': 'subject_driven_models.XFluxModel'
### TRIG DTM models
'sd35_dtm': 'text_to_image_models.SD35DTMModel'
'flux_dtm': 'text_to_image_models.FLUXDTMModel'
'sana_dtm': 'text_to_image_models.SanaDTMModel'
### DTM models with dimension
'sd35_dtm_dim': 'text_to_image_models.SD35DTMDimModel'
'sana_dtm_dim': 'text_to_image_models.SanaDTMDimModel'
'xflux_dtm_dim': 'subject_driven_models.XFluxDTMDimModel'
'hqedit_dtm_dim': 'image_editing_models.HQEditDTMDimModel'
### Multilingual models
'altdiffusion': 'text_to_image_models.AltDiffusionModel'
'mulan': 'text_to_image_models.MuLanModel'
'peadiffusion': 'text_to_image_models.PEADiffusionModel'
