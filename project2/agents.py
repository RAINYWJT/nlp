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



class Agents:
    def __init__(self, api_token: str, base_url: str, model: str, max_retries: int, retry_delay: int, loop: int):
        self.api_token = api_token
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.loop = loop

        self.client = OpenAI(api_key=self.api_token, base_url=self.base_url)

    def chat(self, messages: List[Dict]) -> str:
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"重试中({attempt+1}/{self.max_retries})...错误: {e}")
                time.sleep(self.retry_delay)
        return "无法获取回应"

    def simulate_dialogue(self, case_prompt: str) -> Dict:
        """多智能体对话逻辑"""
        questions = self.extract_questions_from_prompt(case_prompt)
        print(questions)
        # assert 0

        messages = [
            {"role": "system", "content": "你是一个法官，请主持一次关于凶杀案的多方调查会议。请调用侦探和证人的陈述，最终得出结论。"},
            {"role": "user", "content": f"案件描述如下：{case_prompt}。请开始案件调查并主持对话。"}
        ]

        for round in range(self.loop):  # 多轮交互
            reply = self.chat(messages)
            messages.append({"role": "assistant", "content": reply})

            # 侦探提问
            messages.append({"role": "user", "content": "侦探，请根据目前的发言进行分析并质询。"})
            reply = self.chat(messages)
            messages.append({"role": "assistant", "content": reply})

            # 证人回应
            messages.append({"role": "user", "content": "证人，请回答刚才侦探的质询。"})
            reply = self.chat(messages)
            messages.append({"role": "assistant", "content": reply})

        # 最终由法官总结
        messages.append({"role": "user", "content": 
            f"""
            请你作为法官，综合大家的发言，给出最终推理结论：
            请回答以下问题：

            特别注意，要按照格式回答！

            {questions}
            """
        })
                
        final_response = self.chat(messages)
        return {"dialogue": messages, "conclusion": final_response}

    # def extract_solution2(self, text: str) -> dict:
    #     """从文本中提取答案"""
    #     import re
    #     match = re.search(r"(.*?答案.*?)\s*(.*)", text, re.DOTALL)
    #     if not match:
    #         return {}
    #     answer_block = match.group(2).strip()

    #     # 提取 A-K 的答案
    #     matches = re.findall(r'([A-K])\.\s*(\S+)', answer_block)
    #     return {k: v for k, v in matches}
    

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
            model=self.model,  # 如 "gpt-3.5-turbo" 或 "gpt-4"
            messages=messages,
            temperature=0.1  # 低随机性，确保稳定输出
        )

        # 解析 LLM 返回的字典（假设返回的是合法 Python 字典字符串）
        import ast
        try:
            result_dict = ast.literal_eval(response.choices[0].message.content.strip())
            return result_dict
        except (SyntaxError, ValueError):
            # 如果解析失败，手动提取（备用方案）
            return {}
            
    def extract_questions_from_prompt(self, prompt: str) -> str:
        # 使用正则表达式提取 "请回答以下问题" 后的所有内容
        import re
        match = re.search(r"请回答以下问题：(.+)", prompt, re.DOTALL)
        if match:
            return match.group(1).strip()  # 提取并去掉前后多余的空白字符
        else:
            return None  # 如果没有匹配到则返回 None


    def run(self, input_file: str, output_file: str, number: int = -1):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if number > 0:
            data = data[:number]

        results = []
        acc = 0
        require = 0
        optinal = 0

        for item in tqdm(data, desc="多智能体对话中"):
            prompt = item["prompt"]
            solution = item["solution"]

            result = self.simulate_dialogue(prompt)
            print(result["conclusion"])
            # 提取推理结果中的答案
            prediction = self.extract_solution(result["conclusion"], prompt)
            print(prediction)
            
            # 计算准确度
            accuracy = evaluate_response(prediction, solution)
            require_acc = evaluate_response_require(prediction, solution)
            optinal_acc = evaluate_response_optional(prediction, solution)

            acc += accuracy
            require += require_acc
            optinal += optinal_acc

            results.append({
                "prompt": prompt,
                "response": prediction,
                "solution": solution,
                "accuracy": accuracy,
                "req_acc": require_acc,
                "opt_acc": optinal_acc
            })

        summary = {
            "results": results,
            "statistics": {
                "total": len(data),
                "average_accuracy": acc / len(data) if data else 0,
                "average_require": require_acc / len(data) if data else 0,
                "average_optional": optinal_acc / len(data) if data else 0
            }
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"完成模拟！平均准确率：{summary['statistics']['average_accuracy']:.2%}")
