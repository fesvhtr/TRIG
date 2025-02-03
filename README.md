# TRIG
Trade-offs and Relationships in IMage Generation

## Setup
### Installation
```bash
conda create -n trig python=3.10 -y
conda activate trig
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Dataset Zoo
```bash
datasets
├── Trig
│   ├── Trig-image-editing
│   ├── Trig-subject-driven
│   └── Trig-text-to-image
└── raw_datasets
    ├── image-editing
    │   ├── OmniEdit-Filtered
    │   └── X2I-mm-instruction
    ├── subject-driven
    │   ├── Subjects200K-Total
    │   ├── X2I-subject-driven
    │   └── dreambooth
    └── text-to-image
        ├── DOCCI
        └── X2I-text-to-image
```

### Model Zoo
```bash
models
├── BLIP-Diffusion
├── ELITE
├── InstructDiffusion
├── MagicBrush
├── OmniGen
├── One-Diffusion
├── PixArt-Sigma
│   ├── PixArt-Sigma-XL-2-1024-MS
│   └── pixart_sigma_sdxlvae_T5_diffusers
├── Sana
│   ├── Sana_1600M_1024px_BF16_diffusers
│   └── Sana_1600M_1024px_MultiLing_diffusers
├── instruct_pix2pix
└── stable-diffusion
    ├── stable-diffusion-safety-checker
    ├── stable-diffusion-v-1-4-original
    ├── stable-diffusion-v1-4
    ├── stable-diffusion-v1-5
    ├── stable-diffusion-xl-base-1.0
    └── stable-diffusion-xl-refiner-1.0
```


### Inference
Text-to-Image Generation
```bash
python text_to_image.py --model OmniGen
```



