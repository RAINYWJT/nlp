import json
import time
from tqdm import tqdm
from typing import List, Dict
from openai import OpenAI


def evaluate_response(response, solution):
    return 1.0 if response == solution else 0.0

def evaluate_response_require(response, solution):
    response = {k: str(v).strip().lower() for k, v in response.items()}
    solution = {k: str(v).strip().lower() for k, v in solution.items()}
    
    if set(response.keys()) != set(solution.keys()):
        return 0.0
    
    require_keys = list(solution.keys())[:4]
    if not require_keys: 
        return 0.0
    
    correct = sum(1 for key in require_keys if response.get(key) == solution.get(key))
    return correct / len(require_keys)

def evaluate_response_optional(response, solution):
    response = {k: str(v).strip().lower() for k, v in response.items()}
    solution = {k: str(v).strip().lower() for k, v in solution.items()}

    if set(response.keys()) != set(solution.keys()):
        return 0.0

    optional_keys = list(solution.keys())[4:]
    if not optional_keys:  
        return 0.0

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

    def extract_questions_from_prompt(self, prompt: str) -> str:
        # 使用正则表达式提取 "请回答以下问题" 后的所有内容
        import re
        match = re.search(r"请回答以下问题：(.+)", prompt, re.DOTALL)
        if match:
            return match.group(1).strip()  # 提取并去掉前后多余的空白字符
        else:
            return None  # 如果没有匹配到则返回 None

    def extract_solution(self, conclusion_text, case_prompt):
        # 构造 Prompt，让 LLM 提取关键信息并格式化
        questions = self.extract_questions_from_prompt(case_prompt)
        prompt = f"""
        请从以下侦探推理结论中提取关键信息，并严格按照指定格式回答：

        **原始结论：**
        {conclusion_text}

        **要求格式：**
        {questions}

        **要求：**
        1. 只提取关键信息，不要解释推理过程。
        2. 只返回 Python 字典，格式为："A": "格林先生", "B": "喷泉厅", ...。
        3. 严格按以上格式返回答案.
        """

        messages = [
            {"role": "system", "content": "你是一个专业的侦探助手，擅长从文本中提取关键信息并结构化输出。"},
            {"role": "user", "content": prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.model, 
            messages=messages,
            temperature=0.1  
        )

        # 解析 LLM 返回的字典（假设返回的是合法 Python 字典字符串）
        import ast
        try:
            result_dict = ast.literal_eval(response.choices[0].message.content.strip())
            return result_dict
        except (SyntaxError, ValueError):
            return {}

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
                # print(response)
                result = self.extract_solution(response.choices[0].message.content , prompt)


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

