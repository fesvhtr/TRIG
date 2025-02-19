from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from trig.utils.utils import encode_image
from trig.metrics.base import BaseMetric
import torch
from trig.config import gpt_logit_system_msg,gpt_logit_dimension_msg

class TRIGQwen72Metric(BaseMetric):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        # MODEL_PATH = "Qwen/Qwen2.5-VL-72B-Instruct"
        MODEL_PATH = "Benasd/Qwen2.5-VL-72B-Instruct-AWQ"

        self.llm = LLM(
            model=MODEL_PATH,
            port=10021,
            device="cuda",
            tensor_parallel_size=4,
            dtype="bfloat16",
            api="trig",
            limit_mm_per_prompt={"image": 5, "video": 2},
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

        outputs = self.llm.generate([llm_inputs], sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text
        top_logprobs = outputs[0].outputs[0].logprobs.top_logprobs

        score = self.logprobs_score(top_logprobs)
        return round(score, 3)
    
    def logprobs_score(self, top_logprobs):
        score = 0.0

        valid_tokens = ["ex", "excellent", "Excellent", "good", "Good", "medium", "Medium", "bad", "Bad", "terr", "Terr"]

        filtered_tokens = []
        filtered_logprobs = []

        for item in top_logprobs:
            if item.token in valid_tokens:
                filtered_tokens.append(item.token)
                filtered_logprobs.append(float(item.logprob))

        # If there is no matching token, directly return 0 points
        if not filtered_tokens:
            return 0.0

        # Convert log probabilities to linear probabilities, dealing with extreme values
        try:
            linear_probs = [math.exp(lp) for lp in filtered_logprobs]
        except OverflowError:
            linear_probs = [0.0] * len(filtered_logprobs)

        # Renormalise probabilities
        total = sum(linear_probs) + 1e-10
        normalized_probs = [p / total for p in linear_probs]

        for token, prob in zip(filtered_tokens, normalized_probs):
            # print(token, prob)
            if token in ["excellent", "Excellent"]:
                score += 1.0 * prob
            elif token in ["good", "Good"]:
                score += 0.75 * prob
            elif token in ["medium", "Medium"]:
                score += 0.5 * prob
            elif token in ["bad", "Bad"]:
                score += 0.25 * prob
            elif token in ["terr", "Terr"]:
                score += 0 * prob

        return score
            
    def compute_batch(self, data_ids, image_paths, prompts):
        if data_ids is None:
            results = []
            for image_path, prompt in tqdm(zip(images, prompts)):
                results.append(self.compute(image_path, prompt['prompt'])) 
            return results
        
        else:
            results = {}
            for data_id, image_path, prompt in tqdm(zip(data_ids, images, prompts)):
                results[data_id] = self.compute(image_path, prompt['prompt'])
            return results

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