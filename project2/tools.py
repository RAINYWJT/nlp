import json
import time
from tqdm import tqdm
from typing import List, Dict
from openai import OpenAI
import re
from collections import defaultdict

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
        self.client = OpenAI(api_key=self.api_token, base_url=self.base_url)

    # ------------------- 时间线处理核心逻辑 -------------------
    def _parse_time_ranges(self, clues: List[str]) -> Dict[str, List[str]]:
        """从线索中提取人物时间线（示例格式：'格林先生 10:00-12:00 在喷泉厅'）"""
        time_ranges = defaultdict(list)
        time_pattern = re.compile(r"([^\d]+)(\d{1,2}:\d{2})-(\d{1,2}:\d{2})在(.+)")
        
        for clue in clues:
            match = time_pattern.search(clue)
            if match:
                person, start, end, location = match.groups()
                time_ranges[person.strip()].append({
                    "start": self._time_to_minutes(start),
                    "end": self._time_to_minutes(end),
                    "location": location.strip()
                })
        return dict(time_ranges)

    def _time_to_minutes(self, time_str: str) -> int:
        """将时间字符串（如 '10:30'）转换为分钟数（630）"""
        h, m = map(int, time_str.split(':'))
        return h * 60 + m

    def _check_time_conflicts(self, time_ranges: Dict) -> Dict[str, str]:
        """检查时间线冲突并返回可信位置"""
        results = {}
        for person, ranges in time_ranges.items():
            # 按开始时间排序
            sorted_ranges = sorted(ranges, key=lambda x: x["start"])
            
            # 检查连续时间段是否重叠
            for i in range(len(sorted_ranges) - 1):
                if sorted_ranges[i]["end"] > sorted_ranges[i+1]["start"]:
                    raise ValueError(f"{person} 的时间线冲突："
                                   f"{sorted_ranges[i]['location']} 和 {sorted_ranges[i+1]['location']}")
            
            # 取最后一个已知位置（假设最新信息最可靠）
            results[person] = sorted_ranges[-1]["location"]
        return results

    # ------------------- 工具调用流程优化 -------------------
    def logical_solver(self, clues: List[str], question: str) -> Dict:
        """增强版逻辑求解器：优先使用规则引擎"""
        try:
            # 第一步：尝试用时间线引擎
            time_ranges = self._parse_time_ranges(clues)
            if time_ranges:
                time_based_answers = self._check_time_conflicts(time_ranges)
                if "谁在场" in question:  
                    return {"A": list(time_based_answers.keys())[0]}
            
            # 第二步：LLM 兜底处理  
            return self._llm_fallback(clues, question)
        except Exception as e:
            print(f"规则引擎失败，退回LLM: {str(e)}")
            return self._llm_fallback(clues, question)

    def _llm_fallback(self, clues: List[str], question: str) -> Dict:
        """LLM 兜底处理（原 logical_solver 逻辑）"""
        system_msg = """你是一个逻辑谜题专家。严格按以下JSON格式回答：
        {"A": "答案1", "B": "答案2", ...}"""
        
        clues_text = "\n".join(f"- {clue}" for clue in clues)
        prompt = f"线索：\n{clues_text}\n\n问题：{question}"
        
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

    # ------------------- 原有方法（保持不变） -------------------
    def extract_questions_from_prompt(self, prompt: str) -> str:
        match = re.search(r"请回答以下问题：(.+)", prompt, re.DOTALL)
        return match.group(1).strip() if match else None

    def extract_solution(self, conclusion_text: str, case_prompt: str) -> Dict:
        questions = self.extract_questions_from_prompt(case_prompt)
        prompt = f"""从文本提取信息，返回JSON：
        原始文本：{conclusion_text}
        问题：{questions}
        要求：{{"A": "答案1", "B": "答案2", ...}}"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

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
                # Solve the puzzle
                result = self.logical_solver(prompt, question)
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






# class LogicTools:
#     def __init__(self, api_token: str, base_url: str, model: str, max_retries: int = 3, retry_delay: int = 5):
#         self.api_token = api_token
#         self.base_url = base_url
#         self.model = model
#         self.max_retries = max_retries
#         self.retry_delay = retry_delay

#         self.client = OpenAI(
#             api_key=self.api_token,
#             base_url=self.base_url
#         )

    
#     def extract_questions_from_prompt(self, prompt: str) -> str:
#         # 使用正则表达式提取 "请回答以下问题" 后的所有内容
#         import re
#         match = re.search(r"请回答以下问题：(.+)", prompt, re.DOTALL)
#         if match:
#             return match.group(1).strip()  # 提取并去掉前后多余的空白字符
#         else:
#             return None  # 如果没有匹配到则返回 None

#     def extract_solution(self, conclusion_text, case_prompt):
#         # 构造 Prompt，让 LLM 提取关键信息并格式化
#         questions = self.extract_questions_from_prompt(case_prompt)
#         prompt = f"""
#         请从以下侦探推理结论中提取关键信息，并严格按照指定格式回答：

#         **原始结论：**
#         {conclusion_text}

#         **要求格式：**
#         {questions}

#         **要求：**
#         1. 只提取关键信息，不要解释推理过程。
#         2. 只返回 Python 字典，格式为："A": "格林先生", "B": "喷泉厅", ...。
#         3. 严格按以上格式返回答案.
#         """

#         messages = [
#             {"role": "system", "content": "你是一个专业的侦探助手，擅长从文本中提取关键信息并结构化输出。"},
#             {"role": "user", "content": prompt}
#         ]

#         response = self.client.chat.completions.create(
#             model=self.model, 
#             messages=messages,
#             temperature=0.1  
#         )

#         # 解析 LLM 返回的字典（假设返回的是合法 Python 字典字符串）
#         import ast
#         try:
#             result_dict = ast.literal_eval(response.choices[0].message.content.strip())
#             return result_dict
#         except (SyntaxError, ValueError):
#             return {}

#     def extract_clues(self, prompt: str) -> List[str]:
#         """Extract key clues from the puzzle prompt using LLM"""
#         system_msg = """You are a detective assistant that extracts key logical clues from murder mystery puzzles. 
#         Extract ONLY the factual clues, ignoring narrative descriptions. List each clue on a new line."""
        
#         for attempt in range(self.max_retries):
#             try:
#                 response = self.client.chat.completions.create(
#                     model=self.model,
#                     messages=[
#                         {"role": "system", "content": system_msg},
#                         {"role": "user", "content": prompt}
#                     ],
#                     temperature=0.1
#                 )
#                 clues = response.choices[0].message.content.split('\n')
#                 return [clue.strip() for clue in clues if clue.strip()]
#             except Exception as e:
#                 if attempt == self.max_retries - 1:
#                     raise
#                 time.sleep(self.retry_delay)
        
#         return []

#     def logical_solver(self, clues: List[str], question: str) -> Dict:
#         """Solve the logic puzzle using extracted clues"""
#         system_msg = """You are an expert logic puzzle solver. Use the given clues to solve the mystery.
#         Respond with ONLY a JSON object containing the answers to all questions in the format:
#         {"A": "answer1", "B": "answer2", ...}"""
        
#         clues_text = "\n".join(f"- {clue}" for clue in clues)
#         prompt = f"Clues:\n{clues_text}\n\nQuestion: {question}"
        
#         for attempt in range(self.max_retries):
#             try:
#                 response = self.client.chat.completions.create(
#                     model=self.model,
#                     messages=[
#                         {"role": "system", "content": system_msg},
#                         {"role": "user", "content": prompt}
#                     ],
#                     temperature=0.1,
#                     response_format={"type": "json_object"}
#                 )
#                 return json.loads(response.choices[0].message.content)
#             except Exception as e:
#                 if attempt == self.max_retries - 1:
#                     raise
#                 time.sleep(self.retry_delay)
        
#         return {}

#     def run(self, input_file: str, output_file: str, number: int = -1):
#         """Run the complete puzzle solving pipeline"""
#         with open(input_file, "r", encoding="utf-8") as f:
#             data = json.load(f)

#         if number > 0:
#             data = data[:number]

#         results = []
#         acc_total, acc_req, acc_opt = 0, 0, 0
#         success = 0

#         for item in tqdm(data, desc="Solving puzzles"):
#             prompt = item["prompt"]
#             solution = item["solution"]
#             question = item.get("question", "Answer all the questions in the prompt")

#             try:
#                 # Extract key clues
#                 clues = self.extract_clues(prompt)
                
#                 # Solve the puzzle
#                 result = self.logical_solver(clues, question)
#                 result = self.extract_solution(result , prompt)

#                 # Evaluate results
#                 acc = evaluate_response(result, solution)
#                 acc_r = evaluate_response_require(result, solution)
#                 acc_o = evaluate_response_optional(result, solution)

#                 acc_total += acc
#                 acc_req += acc_r
#                 acc_opt += acc_o
#                 success += 1

#                 results.append({
#                     "prompt": prompt,
#                     "clues": clues,
#                     "response": result,
#                     "solution": solution,
#                     "accuracy": acc,
#                     "req_acc": acc_r,
#                     "opt_acc": acc_o
#                 })
#             except Exception as e:
#                 print(f"Error processing item: {e}")
#                 continue

#         summary = {
#             "results": results,
#             "statistics": {
#                 "total": len(data),
#                 "processed": success,
#                 "average_accuracy": acc_total / success if success else 0,
#                 "average_require_accuracy": acc_req / success if success else 0,
#                 "average_optional_accuracy": acc_opt / success if success else 0
#             }
#         }

#         with open(output_file, "w", encoding="utf-8") as f:
#             json.dump(summary, f, ensure_ascii=False, indent=2)

#         print(f"Complete! Processed {success}/{len(data)} puzzles")
#         print(f"Average accuracy: {summary['statistics']['average_accuracy']:.2%}")
#         print(f"Required questions accuracy: {summary['statistics']['average_require_accuracy']:.2%}")
#         print(f"Optional questions accuracy: {summary['statistics']['average_optional_accuracy']:.2%}")

