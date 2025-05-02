import json
import time
from tqdm import tqdm
from typing import List, Dict
from openai import OpenAI
from utils import *

class SyncApiProcessor:
    def __init__(self, api_token: str, base_url: str, model: str, max_retries: int, retry_delay: int):
        if not api_token:
            raise ValueError("API token is required")
        self.api_token = api_token
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        # self.headers = {
        #     "Authorization": f"Bearer {self.api_token}",
        #     "Content-Type": "application/json"
        # }

        self.client = OpenAI(
            api_key=self.api_token,
            base_url=self.base_url,
        )

    def _process_item(self, item: Dict) -> Dict:
        prompt = item.get("prompt", "")
        solution = item.get("solution", {})


        for attempt in range(self.max_retries):
            try:
                # print(body)
                # print(self.base_url)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )

                import re

                def extract_solution_from_response(text):
                    # 找到最后一个包含"答案"字样的句子，并从该句子之后提取答案
                    match = re.search(
                        r'(.*?答案.*?)(.*)', 
                        text, 
                        re.DOTALL
                    )
                    
                    if not match:
                        return {"solution": {}}
                    
                    # 提取答案部分（match.group(2) 是答案部分）
                    answer_block = match.group(2).strip()
                    
                    # 匹配 A. XXX、B. XXX 等项
                    matches = re.findall(r'([A-K])\.\s*(\S+)', answer_block)
                    
                    solution = {key: value for key, value in matches}
                    return solution


                result = extract_solution_from_response(response.choices[0].message.content)
                # print(response_text)

                accuracy = evaluate_response(result, solution)
                require_acc = evaluate_response_require(result, solution)
                optinal_acc = evaluate_response_optional(result, solution)

                return {
                    "prompt": prompt,
                    "response": result,
                    "solution": solution,
                    "accuracy": accuracy, 
                    "req_acc" : require_acc, 
                    "opt_acc" : optinal_acc
                }
            except Exception as e:
                print(f"[Retry {attempt + 1}/{self.max_retries}] Error: {e}")
                time.sleep(self.retry_delay)
        
        return {
            "prompt": prompt,
            "response": "",
            "solution": solution,
            "accuracy": 0.0,
            "req_acc" : 0.0, 
            "opt_acc" : 0.0,
            "error": "Max retries exceeded"
        }

    def run(self, input_file: str, output_file: str, number: int, batch_size: int = 10):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        items = data[:number] if number > 0 else data
        total = len(items)

        results = []
        processed = 0
        total_accuracy = 0.0
        require_acc = 0.0
        optinal_acc = 0.0

        results = []
        processed = 0
        total_accuracy = 0.0
        require_acc = 0.0
        optinal_acc = 0.0

        for item in tqdm(items, total=total, desc="Processing prompts"):
            result = self._process_item(item)
            results.append(result)
            if "error" not in result:
                processed += 1
                total_accuracy += result["accuracy"]
                require_acc += result["req_acc"]
                optinal_acc += result["opt_acc"]


        avg_accuracy = total_accuracy / processed if processed else 0.0
        avg_require_acc = require_acc / processed if processed else 0.0
        avg_optinal_acc = optinal_acc / processed if processed else 0.0

        output = {
            "results": results,
            "statistics": {
                "total_samples_in_file": total,
                "processed_samples_successfully": processed,
                "average_accuracy_on_success": avg_accuracy,
                "average_require_acc": avg_require_acc,
                "average_optinal_acc": avg_optinal_acc,
            },
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"Successfully processed {processed}/{total} prompts")
        print(f"Average accuracy: {avg_accuracy:.2%}")

