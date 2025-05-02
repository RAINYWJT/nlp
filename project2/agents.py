import json
import time
from tqdm import tqdm
from typing import List, Dict
from openai import OpenAI
from utils import *


class Agents:
    def __init__(self, api_token: str, base_url: str, model: str, max_retries: int, retry_delay: int):
        self.api_token = api_token
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.client = OpenAI(api_key=self.api_token, base_url=self.base_url)

    def chat(self, messages: List[Dict]) -> str:
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"重试中({attempt+1}/{self.max_retries})...错误: {e}")
                time.sleep(self.retry_delay)
        return "无法获取回应"

    def simulate_dialogue(self, case_prompt: str) -> Dict:
        """多智能体对话逻辑"""
        messages = [
            {"role": "system", "content": "你是一个法官，请主持一次关于凶杀案的多方调查会议。请调用侦探和证人的陈述，最终得出结论。"},
            {"role": "user", "content": f"案件描述如下：{case_prompt}。请开始案件调查并主持对话。"}
        ]

        for round in range(3):  # 多轮交互
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
        messages.append({"role": "user", "content": "请你作为法官，综合大家的发言，给出最终推理结论：谁是凶手？"})
        final_response = self.chat(messages)
        return {"dialogue": messages, "conclusion": final_response}

    def run(self, input_file: str, output_file: str, number: int = -1):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if number > 0:
            data = data[:number]

        results = []
        acc = 0

        for item in tqdm(data, desc="多智能体对话中"):
            prompt = item["prompt"]
            solution = item["solution"]

            result = self.simulate_dialogue(prompt)

            # 粗略评价准确率
            predicted = {"凶手": None}
            for name in ["A", "B", "C", "D", "E"]:
                if name in result["conclusion"]:
                    predicted["凶手"] = name
                    break

            accuracy = evaluate_response(predicted, solution)
            acc += accuracy

            results.append({
                "prompt": prompt,
                "response": result,
                "solution": solution,
                "accuracy": accuracy
            })

        summary = {
            "results": results,
            "statistics": {
                "total": len(data),
                "average_accuracy": acc / len(data) if data else 0
            }
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"完成模拟！平均准确率：{summary['statistics']['average_accuracy']:.2%}")