# TRIG
Trade-offs and Relationships in IMage Generation

## Setup
### Installation
```bash
conda create -n trig python=3.10 -y
conda activate trig
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Experiment
### TRIG Benchmark
#### Auto Evaluation
1. First Please set up a yaml file in config folder to run an experiment, as the format below:
```yaml
name: "test" # name for this experiment
task: "t2i" # chosen from t2i/p2p/sub, Support one task at a time
# json path for the selected task
prompt_path: "/home/zsc/TRIG/dataset/Trig/Trig-text-to-image/text-to-imgae-new1.json"

generation:
    # selected models
    models: ["flux",]
    
evaluation:
    image_dir: ["data/output/demo",]
    result_dir: "data/result"

dimensions:
    IQ-O:
        metrics: ["GPTLogitMetric"]
    TA-R:
        metrics: ["GPTLogitMetric"]
    TA-S:
        metrics: ["GPTLogitMetric", "AnotherMetric"]
    Other Dimensions:
        metrics: ["OtherMetric"]
```
All the available models & metrics & relation functions could be found [there]().

2. Run the main.py
```
python main.py --config/your_config.yaml
```
3. 
   - Generated images will be saved to data/output/chosen_task/chosen_model
   - Evaluation result will be saved to
   - Relation result will be saved to

#### Manual Evaluation

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





