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

    def _parse_with_llm(self, clues: List[str]) -> Dict:
        """使用LLM解析所有线索，提取结构化时间线和位置信息"""
        system_msg = """你是一个专业的案件分析助手。请将以下线索解析：
        {
            "time_ranges": [
                {
                    "person": "人名",
                    "start": 开始时间(float),
                    "end": 结束时间(float),
                    "location": "地点"
                },
                ...
            ],
            "location_logs": [
                {
                    "person": "人名",
                    "time": 时间点(float),
                    "location": "地点"
                },
                ...
            ]
        }"""
        
        clues_text = "\n".join(f"- {clue}" for clue in clues)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": clues_text}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    def _detect_conflicts_with_llm(self, structured_data: Dict) -> Dict:
        """使用LLM检测所有类型的冲突"""
        system_msg = """请分析以下事件数据，检测并返回所有冲突：
        1. 时间线冲突(同一人时间重叠)
        2. 空间冲突(同一时间不同地点)
        3. 移动可行性(地点间移动时间不足)
        
        返回格式：
        {
            "time_conflicts": [{"person": "人名", "conflict": "冲突描述"}, ...],
            "space_conflicts": [{"time": 时间, "conflict": "冲突描述"}, ...],
            "movement_errors": [{"person": "人名", "error": "错误描述"}, ...],
            "reliable_locations": {"人名": "最可信位置", ...}
        }"""
        
        prompt = json.dumps(structured_data, indent=2, ensure_ascii=False)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    def logical_solver(self, clues: List[str], question: str) -> Dict:
        """智能逻辑求解器：完全基于LLM的解析和冲突检测"""
        try:
            # 第一步：用LLM解析所有线索
            structured_data = self._parse_with_llm(clues)
            
            # 第二步：用LLM检测所有冲突
            conflict_report = self._detect_conflicts_with_llm(structured_data)
            
            # print("冲突检测报告:")
            # print(conflict_report)
            # 如果没有匹配的问题模式，使用LLM直接回答问题
            return self._answer_with_llm(clues, question, conflict_report)
            
        except Exception as e:
            print(f"智能解析失败，退回基础LLM: {str(e)}")
            return self._llm_fallback(clues, question)

    def _answer_with_llm(self, clues: List[str], question: str, conflict_report: Dict = None) -> Dict:
        """使用LLM结合冲突报告回答问题"""
        system_msg = """你是一个侦探助手，已经分析出以下信息：
    {conflict_info}
    请根据分析结果回答问题，返回JSON格式答案：{{"A": "答案1", "B": "答案2", ...}}"""
        
        conflict_info = ""
        if conflict_report:
            conflict_info = "\n".join(
                f"- {k}: {json.dumps(v, ensure_ascii=False)}"
                for k, v in conflict_report.items() if v
            )
        
        clues_text = "\n".join(f"- {clue}" for clue in clues)
        prompt = f"线索：\n{clues_text}\n\n分析结果：\n{conflict_info}\n\n问题：{question}"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg.format(conflict_info=conflict_info)},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

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
        return response.choices[0].message.content

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
