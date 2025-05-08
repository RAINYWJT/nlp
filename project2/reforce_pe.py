import json
import time
from tqdm import tqdm
from openai import OpenAI


def evaluate_response(response, solution):
    response = {k: str(v).strip().lower() for k, v in response.items()}
    solution = {k: str(v).strip().lower() for k, v in solution.items()}
    if response == solution:
        return 1.0
    
    else: 
        return 0

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


class RePromptEngineer:
    def __init__(self, api_token: str, base_url: str, model: str, max_retries: int, retry_delay: int):
        self.api_token = api_token
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = OpenAI(api_key=self.api_token, base_url=self.base_url)
        
        # 新增：经验记忆库
        self.experience_memory = []
        self.max_memory_size = 20  # 保留最近的20个案例
        
        # 策略定义
        self.strategies = {
            "step_by_step": self.prompt_step_by_step,
            "experience_guided": self.prompt_with_experience  # 新增经验引导策略
        }

    def prompt_step_by_step(self, question: str) -> str:
        return f"请一步步推理以下问题，并给出符合逻辑的正确的最终答案：{question}"

    def prompt_with_experience(self, question: str) -> str:
        """新增：结合历史经验的prompt生成"""
        if not self.experience_memory:
            return self.prompt_step_by_step(question)
        
        # 构建经验提示
        experience_prompt = "以下是之前破获的类似案件及正确解法：\n"
        for i, case in enumerate(self.experience_memory[-5:], 1):  # 只取最近的5个案例
            experience_prompt += f"\n案例{i}:\n问题：{case['question']}\n解法：{case['solution']}\n"
        
        return f"""{experience_prompt}
        
        请参考以上案例经验，一步步推理以下新问题，并给出符合逻辑的正确答案：
        {question}"""

    def add_to_memory(self, question: str, solution: dict):
        """新增：将成功案例加入经验库"""
        if len(self.experience_memory) >= self.max_memory_size:
            self.experience_memory.pop(0)  # 移除最旧的案例
        self.experience_memory.append({
            "question": question,
            "solution": solution
        })

    def call_model(self, prompt: str, learning_mode: bool = False) -> str:
        """改进：增加学习模式"""
        for attempt in range(self.max_retries):
            try:
                messages = [{"role": "user", "content": prompt}]
                
                # 学习模式下添加系统提示
                if learning_mode:
                    messages.insert(0, {
                        "role": "system",
                        "content": "你正在学习破案经验，请仔细分析以下案例并记住关键推理模式。"
                    })
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"[Retry {attempt + 1}/{self.max_retries}] Error: {e}")
                time.sleep(self.retry_delay)
        return "[ERROR] Max retries exceeded."

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


    def run(self, input_file: str, output_file: str, number: int = -1):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = data[:number] if number > 0 else data
        results = []
        total = len(items)
        
        accuracy_stats = {strategy_name: {
            "total_accuracy": 0.0,
            "total_require_acc": 0.0,
            "total_optional_acc": 0.0
        } for strategy_name in self.strategies}
        
        processed = {strategy_name: {
            "accuracy": 0,
            "require_acc": 0,
            "optional_acc": 0
        } for strategy_name in self.strategies}

        for idx, item in enumerate(tqdm(items, desc="Prompt Strategies")):
            question = item.get("question") or item.get("prompt")
            if not question:
                continue
                
            ground_truth = item.get("solution", {})
            result = {
                "question": question,
                "strategies": {},
                "solution": ground_truth
            }

            # 先使用基础策略
            base_strategy = "step_by_step"
            prompt = self.strategies[base_strategy](question)
            response = self.call_model(prompt)
            prediction = self.extract_solution(response, prompt)
            
            # 评估并学习
            accuracy = evaluate_response(prediction, ground_truth)
            if accuracy > 0.1:  # 只有高准确率的案例才加入经验库 Ps：0.1可能又有效，但是我没钱了
                self.add_to_memory(question, ground_truth)
                
                # 学习模式：让模型明确学习这个案例
                learning_prompt = f"""请学习以下破案经验：
                案件描述：{question}
                正确解法：{ground_truth}
                请总结这个案例的关键推理步骤。"""
                self.call_model(learning_prompt, learning_mode=True)

            # 使用所有策略（包括经验引导策略）
            for strategy_name, strategy_fn in self.strategies.items():
                prompt = strategy_fn(question)
                response = self.call_model(prompt)
                prediction = self.extract_solution(response, prompt)

                # 评估和统计（保持原有逻辑）
                accuracy = evaluate_response(prediction, ground_truth)
                accuracy_stats[strategy_name]["total_accuracy"] += accuracy
                processed[strategy_name]["accuracy"] += 1

                require_acc = evaluate_response_require(prediction, ground_truth)
                optional_acc = evaluate_response_optional(prediction, ground_truth)
                
                accuracy_stats[strategy_name]["total_require_acc"] += require_acc
                accuracy_stats[strategy_name]["total_optional_acc"] += optional_acc
                
                processed[strategy_name]["require_acc"] += 1 if require_acc > 0 else 0
                processed[strategy_name]["optional_acc"] += 1 if optional_acc > 0 else 0

                result["strategies"][strategy_name] = {
                    "prompt": prompt,
                    "response": response,
                    "parsed": prediction,
                    "accuracy": accuracy,
                    "req_acc": require_acc,
                    "opt_acc": optional_acc
                }

            results.append(result)

        # 保存结果和统计信息（保持原有逻辑）
        output = {
            "results": results,
            "statistics": {
                "total_samples": total,
                "processed_samples": len(results),
                "experience_memory_size": len(self.experience_memory),
                "strategy_averages": {
                    name: {
                        "avg_accuracy": stats["total_accuracy"] / processed[name]["accuracy"],
                        "avg_require": stats["total_require_acc"] / max(1, processed[name]["require_acc"]),
                        "avg_optional": stats["total_optional_acc"] / max(1, processed[name]["optional_acc"])
                    } for name, stats in accuracy_stats.items()
                }
            }
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)