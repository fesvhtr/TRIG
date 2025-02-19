# TRIG
Trade-offs and Relationships in Image Generation: How Do Different Evaluation Dimensions Interact?

## About
Here is a fast and easy-to-use library for image generation model inference and evaluation.


## Setup
### Installation
```bash
conda create -n trig python=3.10 -y
conda activate trig
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```
We recommand to use TRIG evaluation toolset by [vllm](https://github.com/vllm-project/vllm). Please install with
```
# for Qwen2.5vl, please update your transformers
pip install git+https://github.com/huggingface/transformers@f3f6c86582611976e72be054675e2bf0abb5f775
pip install accelerate
pip install 'vllm>=0.7.2'
```
Then deploy the selected VLM models, currently the TRIG score support GPT series, Qwen2.5-VL series, and LLaVA-NeXT Series. For more information, please visit vllm document.
```
# use Qwen2.5-VL-7B
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 --device cuda --host 0.0.0.0 --dtype bfloat16 --limit-mm-per-prompt image=5,video=5

# or use Qwen2.5-VL-72B quantize version to reduce memory usage
vllm serve Benasd/Qwen2.5-VL-72B-Instruct-AWQ --dtype float16 --port 8000 --gpu-memory-utilization 0.85 --tensor-parallel-size 2 --quantization awq --limit_mm_per_prompt image=4
```

## Getting Started
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

2. Run the eval.py
```
python eval.py --config your_config.yaml

python eval.py --config config/relation.yaml   
```
3. 
   - Generated images will be saved to data/output/your_task/your_model
   - Evaluation result will be saved to 
   - Relation result will be saved to 

#### Manual Evaluation

### Metric Tool Set
## Support List
### Metric Zoo
**A. Omni Metric**  
**TRIG Score**

**B. General Metric**
1. CLIPScore

**C. Specific Metric**  
Image Quality - Relism
1. FID ()
2. IS ()
3. dasdad ()

Image Quality - Originality
1. FID ()
2. IS ()
3. dasdad ()

Image Quality - Aesthetics
1. FID ()
2. IS ()
3. dasdad ()

Task Alignment - Content Alignment
1. FID ()
2. IS ()
3. dasdad ()
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



### Citation

CUDA_VISIBLE_DEVICES=2,3 vllm serve Benasd/Qwen2.5-VL-72B-Instruct-AWQ --dtype float16 --port 4701 --gpu-memory-utilization 0.85 --quantization awq --limit_mm_per_prompt image=4
nohup bash -c "CUDA_VISIBLE_DEVICES=2,3 vllm serve Benasd/Qwen2.5-VL-72B-Instruct-AWQ --dtype float16 --port 8000 --gpu-memory-utilization 0.85 --tensor-parallel-size 2 --quantization awq --limit_mm_per_prompt image=4 " > server2.log 2>&1 &
curl http://localhost:10021/v1/models
curl http://localhost:8000/v1/models
nohup python eval.py > eval2.log 2>&1 &
