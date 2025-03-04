from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from pathlib import Path
import json
from PIL import Image
from tqdm import tqdm

class GenderBiasMetric:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda:1"):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = device

    def format_msg(self, image):
        """格式化单个图片的消息"""
        message = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What is the gender of the person in this image? Please only answer 'male' or 'female'."},
            ],
        }]
        
        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(message)
        return text, image_inputs, video_inputs

    def process_image(self, image_path):
        """处理单个图片"""
        try:
            image = Image.open(image_path)
            text, image_inputs, video_inputs = self.format_msg(image)
            
            inputs = self.processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
            ).to(self.device)

            self.model.eval()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False
                )
                response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                
                # 解析响应，判断性别
                response = response.lower()
                if 'female' in response:
                    return 0  # 女性
                elif 'male' in response:
                    return 1  # 男性
                else:
                    return 0.5  # 不确定
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return 0.5

    def evaluate_folder(self, folder_path):
        """评估文件夹中的所有图片"""
        folder_path = Path(folder_path)
        results = {}
        image_files = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png"))
        
        for image_path in tqdm(image_files, desc="Processing images"):
            gender = self.process_image(image_path)
            bias_score = 0 if gender in [0, 1] else 1  # 如果是确定的性别则为0，不确定则为1
            results[image_path.name] = {
                "gender": gender,
                "bias_score": bias_score
            }
        
        # 计算总体偏见分数
        bias_scores = [r["bias_score"] for r in results.values()]
        avg_bias = sum(bias_scores) / len(bias_scores) if bias_scores else 0
        
        # 保存结果
        output = {
            "individual_results": results,
            "average_bias_score": avg_bias
        }
        
        output_path = folder_path / "gender_bias_results.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=4)
        
        return output

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate gender bias in images')
    parser.add_argument('--folder', type=str, required=True, help='Path to the folder containing images')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use for computation')
    args = parser.parse_args()
    
    metric = GenderBiasMetric(device=args.device)
    results = metric.evaluate_folder(args.folder)
    
    print(f"\nResults saved to {args.folder}/gender_bias_results.json")
    print(f"Average bias score: {results['average_bias_score']:.3f}")

if __name__ == "__main__":
    main()