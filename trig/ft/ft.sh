export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export WANDB_API_KEY="da3ef2608ceaa362d6e40d1d92b4e4e6ebbe9f82"
export CUDA_VISIBLE_DEVICES=2,3
accelerate launch --config_file ./accelerate.yaml train_lora_flux_advanced.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name="/home/muzammal/Projects/TRIG/trig/ft/flux_ft_ds" \
  --output_dir="/home/muzammal/Projects/TRIG/trig/ft/flux_ft" \
  --image_column="image" \
  --caption_column="prompt" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=2 \
  --repeats=1 \
  --report_to="wandb" \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --learning_rate=4e-4 \
  --text_encoder_lr=5e-6 \
    --train_text_encoder \
  --optimizer="AdamW" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --rank=8 \
  --num_train_epochs=10 \
  --checkpointing_steps=206 \
  --seed="42" \


# CUDA_VISIBLE_DEVICES=2,3 python test_lora_flux_advanced.py