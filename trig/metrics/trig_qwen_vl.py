from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from trig.config import gpt_logit_system_msg,gpt_logit_dimension_msg
from tqdm import tqdm


class TRIGQwenMetric(BaseMetric):
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda:1"):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            # remove the following line if don't want to use flash_attention_2
            attn_implementation="flash_attention_2",
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        # The default range for the number of visual tokens per image in the model is 4-16384.
        # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
        # min_pixels = 256*28*28
        # max_pixels = 1280*28*28
        # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
        self.device = device

    def format_msg(self, images, prompts):
        messages = []
        for image, prompt in zip(images, prompts):
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ])
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        return texts, image_inputs, video_inputs


    def compute(self, image, question):
        pass

    def compute_batch(self, data_ids, images, prompts):
        if data_ids is None:
            pass
        else:
            for data_id, image, prompt in zip(data_ids, images, prompts):
                texts, image_inputs, video_inputs = self.format_msg(images, prompts)
                # Preparation for batch inference
                inputs = processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda:1")

                model.eval()
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    last_token_logits = logits[:, -1, :]
                    predicted_token_ids = torch.argmax(last_token_logits, dim=-1)
                    predicted_tokens = [processor.tokenizer.decode(token_id) for token_id in predicted_token_ids]
                    print("Predicted tokens:", predicted_tokens)


if __name__ == "__main__":
    metric = TRIGQwenMetric()
    data_ids = [1, 2]
    images = ["https://openai.com/research/learning-to-follow-instructions/", "https://openai.com/research/learning-to-follow-instructions/"]
    prompts = ["Is this a building in this image?", "Is this a building in this image?"]
    metric.compute_batch(data_ids, images, prompts)