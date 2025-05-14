import json
import time
from tqdm import tqdm
from openai import OpenAI


def evaluate_response(response, solution):
    # 统一小写并去除空格
    response = {k: str(v).strip().lower() for k, v in response.items()}
    solution = {k: str(v).strip().lower() for k, v in solution.items()}

    # 判断是否完全匹配
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



class PromptEngineer:
    def __init__(self, api_token: str, base_url: str, model: str, max_retries: int, retry_delay: int):
        self.api_token = api_token
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.client = OpenAI(
            api_key=self.api_token,
            base_url=self.base_url,
        )

        self.strategies = {
            "sherlock": self.prompt_sherlock,
            "step_by_step": self.prompt_step_by_step,
        }

    def prompt_sherlock(self, question: str) -> str:
        return f"你是福尔摩斯，接到一个案子：{question} 请详细推理并找出答案。"

    def prompt_step_by_step(self, question: str) -> str:
        return f"请一步步推理以下问题，并给出符合逻辑的正确的最终答案：{question}"

    def call_model(self, prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
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

        # 初始化统计字典，每个策略的方法单独存储统计
        accuracy_stats = {strategy_name: {"total_accuracy": 0.0, "total_require_acc": 0.0, "total_optional_acc": 0.0}
                        for strategy_name in self.strategies}
        
        processed = {strategy_name: {"accuracy": 0, "require_acc": 0, "optional_acc": 0}
                    for strategy_name in self.strategies}

        for idx, item in enumerate(tqdm(items, desc="Prompt Strategies")):
            question = item.get("question") or item.get("prompt")
            if not question:
                print(f"[Warning] Missing question in item #{idx}, skipping.")
                continue
            ground_truth = item.get("solution", {})

            result = {
                "question": question,
                "strategies": {},
                "solution": ground_truth
            }

            # 累计每个策略的准确度
            for strategy_name, strategy_fn in self.strategies.items():
                prompt = strategy_fn(question)
                response = self.call_model(prompt)
                prediction = self.extract_solution(response, prompt)

                # 使用 evaluate_response 方法计算准确度
                accuracy = evaluate_response(prediction, ground_truth)
                accuracy_stats[strategy_name]["total_accuracy"] += accuracy
                processed[strategy_name]["accuracy"] += 1

                # 使用 evaluate_response_require 和 evaluate_response_optional 方法计算准确度
                require_acc_value = evaluate_response_require(prediction, ground_truth)
                optional_acc_value = evaluate_response_optional(prediction, ground_truth)

                accuracy_stats[strategy_name]["total_require_acc"] += require_acc_value
                accuracy_stats[strategy_name]["total_optional_acc"] += optional_acc_value

                processed[strategy_name]["require_acc"] += 1 if require_acc_value > 0 else 0
                processed[strategy_name]["optional_acc"] += 1 if optional_acc_value > 0 else 0

                result["strategies"][strategy_name] = {
                    "prompt": prompt,
                    "response": response,
                    "parsed": prediction,
                    "accuracy": accuracy,
                    "req_acc": require_acc_value,
                    "opt_acc": optional_acc_value
                }

            results.append(result)

        # 计算不同方法的平均准确度
        strategy_averages = {}
        for strategy_name in self.strategies:
            avg_accuracy = accuracy_stats[strategy_name]["total_accuracy"] / processed[strategy_name]["accuracy"] if processed[strategy_name]["accuracy"] else 0.0
            avg_require_acc = accuracy_stats[strategy_name]["total_require_acc"] / processed[strategy_name]["require_acc"] if processed[strategy_name]["require_acc"] else 0.0
            avg_optional_acc = accuracy_stats[strategy_name]["total_optional_acc"] / processed[strategy_name]["optional_acc"] if processed[strategy_name]["optional_acc"] else 0.0

            strategy_averages[strategy_name] = {
                "average_accuracy": avg_accuracy,
                "average_require_acc": avg_require_acc,
                "average_optional_acc": avg_optional_acc
            }

        # 添加统计信息到输出
        output = {
            "results": results,
            "statistics": {
                "total_samples_in_file": total,
                "processed_samples_successfully": sum(processed[strategy]["accuracy"] for strategy in self.strategies),
                "strategies_averages": strategy_averages,
            },
        }

        # 保存结果到文件
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        # 打印处理结果
        print(f"Processed {len(results)} samples with {len(self.strategies)} strategies.")
        for strategy_name, averages in strategy_averages.items():
            print(f"Strategy: {strategy_name}")
            print(f"Average accuracy: {averages['average_accuracy']:.2%}")
            print(f"Average require accuracy: {averages['average_require_acc']:.2%}")
            print(f"Average optional accuracy: {averages['average_optional_acc']:.2%}")
