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



class LogicTools:
    def __init__(self, api_token: str, base_url: str, model: str, max_retries: int = 3, retry_delay: int = 5):
        self.api_token = api_token
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.client = OpenAI(
            api_key=self.api_token,
            base_url=self.base_url
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

    def extract_clues(self, prompt: str) -> List[str]:
        """Extract key clues from the puzzle prompt using LLM"""
        system_msg = """You are a detective assistant that extracts key logical clues from murder mystery puzzles. 
        Extract ONLY the factual clues, ignoring narrative descriptions. List each clue on a new line."""
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                clues = response.choices[0].message.content.split('\n')
                return [clue.strip() for clue in clues if clue.strip()]
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_delay)
        
        return []

    def logical_solver(self, clues: List[str], question: str) -> Dict:
        """Solve the logic puzzle using extracted clues"""
        system_msg = """You are an expert logic puzzle solver. Use the given clues to solve the mystery.
        Respond with ONLY a JSON object containing the answers to all questions in the format:
        {"A": "answer1", "B": "answer2", ...}"""
        
        clues_text = "\n".join(f"- {clue}" for clue in clues)
        prompt = f"Clues:\n{clues_text}\n\nQuestion: {question}"
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_delay)
        
        return {}

    def run(self, input_file: str, output_file: str, number: int = -1):
        """Run the complete puzzle solving pipeline"""
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if number > 0:
            data = data[:number]

        results = []
        acc_total, acc_req, acc_opt = 0, 0, 0
        success = 0

        for item in tqdm(data, desc="Solving puzzles"):
            prompt = item["prompt"]
            solution = item["solution"]
            question = item.get("question", "Answer all the questions in the prompt")

            try:
                # Extract key clues
                clues = self.extract_clues(prompt)
                
                # Solve the puzzle
                result = self.logical_solver(clues, question)
                result = self.extract_solution(result , prompt)

                # Evaluate results
                acc = evaluate_response(result, solution)
                acc_r = evaluate_response_require(result, solution)
                acc_o = evaluate_response_optional(result, solution)

                acc_total += acc
                acc_req += acc_r
                acc_opt += acc_o
                success += 1

                results.append({
                    "prompt": prompt,
                    "clues": clues,
                    "response": result,
                    "solution": solution,
                    "accuracy": acc,
                    "req_acc": acc_r,
                    "opt_acc": acc_o
                })
            except Exception as e:
                print(f"Error processing item: {e}")
                continue

        summary = {
            "results": results,
            "statistics": {
                "total": len(data),
                "processed": success,
                "average_accuracy": acc_total / success if success else 0,
                "average_require_accuracy": acc_req / success if success else 0,
                "average_optional_accuracy": acc_opt / success if success else 0
            }
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"Complete! Processed {success}/{len(data)} puzzles")
        print(f"Average accuracy: {summary['statistics']['average_accuracy']:.2%}")
        print(f"Required questions accuracy: {summary['statistics']['average_require_accuracy']:.2%}")
        print(f"Optional questions accuracy: {summary['statistics']['average_optional_accuracy']:.2%}")

