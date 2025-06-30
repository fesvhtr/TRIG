# TRIG
Trade-offs and Relationships in Image Generation: How Do Different Evaluation Dimensions Interact?

## TODO

1. [x] v0: Release the TRIG dataset and evaluation pipeline.
2. [x] v1: Release the Finetune pipeline and experiments.
3. [ ] v1: Release the additional metrics toolkit.
3. [ ] v1: Release the RL (DDPO) pipeline and experiments.
4. [ ] v1.5: 
5. [ ] v2:

## Setup
### Installation
```bash
conda create -n trig python=3.10 -y
conda activate trig
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```
We recommand to use TRIG score by [vllm](https://github.com/vllm-project/vllm). Please install with
```
# for Qwen2.5vl, please update your transformers
pip install git+https://github.com/huggingface/transformers@f3f6c86582611976e72be054675e2bf0abb5f775
pip install accelerate
pip install 'vllm>=0.7.2'
# vllm==0.7.3 and torch=2.6.0 work well
```
Then deploy the selected VLM models, currently the TRIG score support GPT series, Qwen2.5-VL series, and LLaVA-NeXT Series. For more information, please visit vllm document.
```
# use Qwen2.5-VL-7B
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 --device cuda --host 0.0.0.0 --dtype bfloat16 --limit-mm-per-prompt image=5,video=5

# or use Qwen2.5-VL-72B with quantize version
vllm serve Qwen/Qwen2-VL-72B-Instruct-AWQ --dtype float16 --port 8000 --gpu-memory-utilization 0.85 --tensor-parallel-size 2 --quantization awq --limit_mm_per_prompt image=4
```

## Getting Started
### Auto Evaluation pipeline on TRIG Benchmark
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
    Other Dimensions You Want:
        metrics: ["OtherMetric"]

relation:
  models: ["flux"]
  res: "formatted_flux"
  metric: "spearman_corr"
  plot: true
  heatmap: true
  tsne: true
  tradeoff: true
  quadrant_analysis: true
  thresholds:
    synergy: 0.8 
    bottleneck: 0.5 
    
  insight_thresholds:
    synergy_density: 0.4
    bottleneck_density: 0.4
    dominance_ratio: 0.8
    tradeoff_corr: 0.6
```
More examples could be found in the config folder.

2. Run ```eval.py```
```
python eval.py --config your_config.yaml
```
3. Outputs:
- Generated images will be saved to ```data/output/your_task/your_model/```
- Evaluation result will be saved to ```data/output/your_task/your_model.json```
- Relation result will be saved to ```data/output/your_model/```
4. Notes:
TBD

### Manual Evaluation by metrics toolkit
All the metrics could be used **independently**. For example:
```
metric_class = trig.metrics.import_metric("aesthetic_predictor")
metric_instance = metric_class()
# Single Evaluation
score = metric_instance.compute(image_path="/path/to/image", prompt="prompt")
# Batch Evaluation
score = metric_instance.compute_batch_manual(images=["/path/to/image"], prompts=["prompt"])
```


### Finetuning by TRIG Result
TBD
### RL (DDPO) on SD 1.5
TBD
### DTM (Dimension Trade-off Map)
TBD
## Support List
[Model Zoo]()  
[Metric Zoo]()

## Acknowledgement
TBD
## Citation
```
TBD
```