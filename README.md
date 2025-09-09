# TRIG
[![paper](https://img.shields.io/badge/cs.CV-2507.22100-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2507.22100)
[![Dataset](https://img.shields.io/badge/Dataset-TRIG-orange)](https://huggingface.co/datasets/TRIG-bench/TRIG)
[![Collection](https://img.shields.io/badge/Collection-Download-blue)](https://huggingface.co/collections/TRIG-bench/trig-6862b38b91af9bec3a4a05cb)   
Trade-offs and Relationships in Image Generation: How Do Different Evaluation Dimensions Interact?

## TODO

1. [x] v0: Release the TRIG dataset and evaluation pipeline.
2. [x] v1: Release the Finetune pipeline and experiments.
3. [ ] v1: Release the RL (DDPO) pipeline and experiments.
4. [ ] v1.5: TBD.
5. [ ] v2: TBD.

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

2. Run ```main.py```
```
python main.py --config your_config.yaml
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


### Finetuning by DTM
1. Select the dimension and trade-off type you want to optimize. for example, in the paper, we choose Knowledge & Ambiguity, and try to balance these two dimensions.
2. Follow the TRIG principle, we create [a original set](https://huggingface.co/datasets/TRIG-bench/flux-ft-ds) which covers the two dim.
3. We generate images with this set, the ouput images are in [flux_ft_train.zip](https://huggingface.co/datasets/TRIG-bench/flux_ft_train/blob/main/flux_ft_train.zip).
4. Test these images, select [good samples](https://huggingface.co/datasets/TRIG-bench/flux_ft_train/blob/main/flux_ft_72B_filtered_ids.json) with trade-off as expected.
5. Use these selected image to do LoRA finetune on flux.
6. Then we got the [balanced flux model](https://huggingface.co/TRIG-bench/FLUX_FT_LoRA_TRIG_epoch10).

### Prompt Engineering by DTM
use model name 'sd35_dtm_dim', 'sana_dtm_dim', 'xflux_dtm_dim' and 'hqedit_dtm_dim' in the yaml config file to generate with Prompt Engineering.

## Acknowledgement
Many thanks to the great works in GenAI Models like [FLUX](https://huggingface.co/black-forest-labs/FLUX.1-dev), Benchmarks like [HEIM](https://crfm.stanford.edu/helm/heim/latest/), Metric like [VQAScore](https://github.com/linzhiqiu/t2v_metrics).

## Citation
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
```
@article{zhang2025trade,
  title={Trade-offs in Image Generation: How Do Different Dimensions Interact?},
  author={Zhang, Sicheng and Xie, Binzhu and Yan, Zhonghao and Zhang, Yuli and Zhou, Donghao and Chen, Xiaofei and Qiu, Shi and Liu, Jiaqi and Xie, Guoyang and Lu, Zhichao},
  journal={arXiv preprint arXiv:2507.22100},
  year={2025}
}
```
