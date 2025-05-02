import json
import time
from tqdm import tqdm
from typing import List, Dict
from openai import OpenAI
from utils import *


class LogicTools:
    def __init__(self, api_token: str, base_url: str, model: str, max_retries: int, retry_delay: int):
        self.api_token = api_token
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.client = OpenAI(
            api_key=self.api_token,
            base_url=self.base_url
        )

    def extract_clues(self, prompt: str) -> List[str]:
        """使用模型提取线索句"""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": f"请将以下文本中的每个人的发言提取为 JSON 格式列表，例如：['A: ...', 'B: ...']：\n\n{prompt}"
                    }],
                    temperature=0
                )
                text = response.choices[0].message.content
                clues = self.parse_response_to_clues(text)
                return clues
            except Exception as e:
                print(f"提取线索失败，重试 {attempt + 1}/{self.max_retries}：{e}")
                time.sleep(self.retry_delay)
        return []

    def parse_response_to_clues(self, text: str) -> List[str]:
        """简单解析成 ["A: ...", "B: ..."] 格式"""
        lines = text.strip().split("\n")
        return [line.strip("“”\"") for line in lines if ":" in line]

    def logical_solver(self, clues: List[str]) -> Dict:
        """核心逻辑推理工具：只有一个人说真话"""
        people = ['A', 'B', 'C', 'D', 'E']
        suspects = set(p[0] for p in clues if ":" in p)

        for killer in suspects:
            truth_tellers = 0
            for clue in clues:
                speaker, statement = clue.split(":", 1)
                speaker = speaker.strip()
                statement = statement.strip()

                is_truth = False
                if "不是" in statement and killer in statement:
                    is_truth = False
                elif "是" in statement and killer in statement:
                    is_truth = True
                elif f"{killer}不是凶手" in statement:
                    is_truth = False
                elif f"{killer}是凶手" in statement:
                    is_truth = True

                if is_truth:
                    truth_tellers += 1

            if truth_tellers == 1:
                return {"凶手": killer}

        return {"凶手": "无法判断"}

    def run(self, input_file: str, output_file: str, number: int = -1):
        """运行完整流程"""
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if number > 0:
            data = data[:number]

        results = []
        acc_total, acc_req, acc_opt = 0, 0, 0
        success = 0

        for item in tqdm(data, desc="使用工具进行推理"):
            prompt = item["prompt"]
            solution = item["solution"]

            clues = self.extract_clues(prompt)
            result = self.logical_solver(clues)

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

        print(f"完成！共处理 {success}/{len(data)} 条，准确率：{summary['statistics']['average_accuracy']:.2%}")