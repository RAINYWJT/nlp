import json
import time
from tqdm import tqdm
from utils import *
from openai import OpenAI


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
        return f"请一步步推理以下问题，并给出最终答案：{question}"

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

    def extract_solution(self, text: str) -> dict:
        import re
        match = re.search(r"(.*?答案.*?)\s*(.*)", text, re.DOTALL)
        if not match:
            return {}
        answer_block = match.group(2).strip()
        matches = re.findall(r'([A-K])\.\s*(\S+)', answer_block)
        return {k: v for k, v in matches}

    def run(self, input_file: str, output_file: str, number: int = -1):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = data[:number] if number > 0 else data
        results = []
        total = len(items)

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

            for strategy_name, strategy_fn in self.strategies.items():
                prompt = strategy_fn(question)
                response = self.call_model(prompt)
                prediction = self.extract_solution(response)
                acc = evaluate_response(prediction, ground_truth)

                result["strategies"][strategy_name] = {
                    "prompt": prompt,
                    "response": response,
                    "parsed": prediction,
                    "accuracy": acc
                }

            results.append(result)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Processed {len(results)} samples with {len(self.strategies)} strategies.")
