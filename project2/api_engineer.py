import json
import time
from tqdm import tqdm
from typing import List, Dict
from openai import OpenAI


def evaluate_response(response, solution):
    # print(response, solution)
    """Evaluate the match between the model response and the standard answer."""
    # response_lower = response.lower()
    # expected_answers = [f"{key}. {value}".lower() for key, value in solution.items()]
    # all_correct = all(expected_answer in response_lower for expected_answer in expected_answers)
    # accuracy = 1.0 if all_correct else 0.0
    # return accuracy
    return 1.0 if response == solution else 0.0

def evaluate_response_require(response, solution):
    # response_lower = response.lower()
    # expected_answers = [f"{key}. {value}".lower() for key, value in solution.items()]
    # expected_answers = expected_answers[:4]
    # correct_count = sum(1 for expected_answer in expected_answers if expected_answer in response_lower)
    # accuracy = correct_count / len(expected_answers) if expected_answers else 0.0
    # return accuracy
    response = {k: str(v).strip().lower() for k, v in response.items()}
    solution = {k: str(v).strip().lower() for k, v in solution.items()}
    
    # 检查键是否一致
    if set(response.keys()) != set(solution.keys()):
        return 0.0
    
    # 提取前4项的键
    require_keys = list(solution.keys())[:4]
    if not require_keys:  # 避免空列表
        return 0.0
    
    # 计算匹配数
    correct = sum(1 for key in require_keys if response.get(key) == solution.get(key))
    return correct / len(require_keys)

def evaluate_response_optional(response, solution):
    # response_lower = response.lower()
    # expected_answers = [f"{key}. {value}".lower() for key, value in solution.items()]
    # expected_answers = expected_answers[4:]
    # correct_count = sum(1 for expected_answer in expected_answers if expected_answer in response_lower)
    # accuracy = correct_count / len(expected_answers) if expected_answers else 0.0
    # return accuracy
    response = {k: str(v).strip().lower() for k, v in response.items()}
    solution = {k: str(v).strip().lower() for k, v in solution.items()}
    
    # 检查键是否一致
    if set(response.keys()) != set(solution.keys()):
        return 0.0
    
    # 提取剩余项的键
    optional_keys = list(solution.keys())[4:]
    if not optional_keys:  # 避免空列表
        return 0.0
    
    # 计算匹配数
    correct = sum(1 for key in optional_keys if response.get(key) == solution.get(key))
    return correct / len(optional_keys)


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

