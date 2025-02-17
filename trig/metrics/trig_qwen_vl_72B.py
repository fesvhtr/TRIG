from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from trig.utils.utils import encode_image
from trig.metrics.base import BaseMetric
import torch
from trig.config import gpt_logit_system_msg,gpt_logit_dimension_msg

class TRIGQwen72Metric(BaseMetric):
    def __init__(self):
        MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

        self.llm = LLM(
            model=MODEL_PATH,
            limit_mm_per_prompt={"image": 10, "video": 10},
        )
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=256,
            stop_token_ids=[],
            logprobs=5,
        )
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH)

    def format_msg(self, prompt, image):
        sys_msg = [{
            "role": "system",
            "content": gpt_logit_system_msg.format(gpt_logit_dimension_msg[self.dimension])
        }]
        user_msg = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image",
                "image": {"url": f"data:image/{image['type']};base64,{image['base64']}"}}]
        }]
        return sys_msg + user_msg

    def compute(self, image_path, prompt):
        image = encode_image(image_path)
        messages = self.format_msg(prompt, image)
        
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,

            # FPS will be returned in video_kwargs
            "mm_processor_kwargs": video_kwargs,
        }

        outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text
        top_logprobs = outputs[0].outputs[0].logprobs.top_logprobs

        print(generated_text)
        print("Top 5 logprobs per token:", top_logprobs)
    
    def compute_batch(self, data_ids, image_paths, prompts):
        pass

if __name__ == "__main__":
    metric = TRIGQwen72Metric()
    image_path = r"/home/muzammal/Projects/TRIG/demo.jpg"
    prompt = ["A old building like a main building of a university",
              "A old building like a main building of a university with green grass and blue sky",
              "A dog swimming in the water",
              "A new techinical building with a grey sky",
              "A beautiful sunset with a beach"]

    for i in prompt:
        result = metric.compute(image_path, i)
        print(result)