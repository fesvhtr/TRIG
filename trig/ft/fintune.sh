export MODEL_NAME="/home/yanzhonghao/data/model_hub/FLUX.1-dev"
export DATASET_NAME="/home/yanzhonghao/data/dataset_hub/3d_icon"
export OUTPUT_DIR="/home/yanzhonghao/data/experiments/3d-icon-Flux-LoRA"
export WANDB_API_KEY="8910b96be5c09b0604a0e905be81ec0463f4828a"

accelerate launch --config_file ./accelerate.yaml train_lora_flux_advanced.py.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --output_dir=$OUTPUT_DIR \
  --caption_column="prompt" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=2 \
  --repeats=1 \
  --report_to="wandb" \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1.0 \
  --text_encoder_lr=1.0 \
  --optimizer="prodigy" \
  --train_text_encoder \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --rank=8 \
  --max_train_steps=700 \
  --checkpointing_steps=2000 \
  --seed="0" \


CUDA_VISIBLE_DEVICES=2 python test_lora_flux_advanced.py