import json
import time
import requests
import argparse
from tqdm import tqdm
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from utils import *


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

    def _process_item(self, item: Dict) -> Dict:
        prompt = item.get("prompt", "")
        solution = item.get("solution", {})


        for attempt in range(self.max_retries):
            try:
                body = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "temperature": 0.7,
                }

                # print(body)
                # print(self.base_url)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                response_text = response.choices[0].message.content

                accuracy = evaluate_response(response_text, solution)
                require_acc = evaluate_response_require(response_text, solution)
                optinal_acc = evaluate_response_optional(response_text, solution)

                return {
                    "prompt": prompt,
                    "response": response_text,
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

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {executor.submit(self._process_item, item): item for item in items}
            for future in tqdm(as_completed(futures), total=total, desc="Processing prompts"):
                result = future.result()
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



def main():
    parser = argparse.ArgumentParser(description="Simplified API Processor (sync version)")
    parser.add_argument("--method",type=int, default=0, help="Number of prompts to process: (0: API),(1: Prompt Engineer),(2:Tools),(3:Agent)")


    parser.add_argument("-n", "--number", type=int, default=0, help="Number of prompts to process (0 for all)")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retry times")
    parser.add_argument("--retry-delay", type=int, default=1, help="Initial delay between retries (seconds)")
   
    parser.add_argument("--input-file", type=str, default='data/tc_200_zh.json', help="Input file")
    parser.add_argument("--output-file", type=str, default='output/tc_200_zh_output.json', help="Output file")
    
    # free QWQ
    # parser.add_argument("--api-token", type=str, default="sk-hlYe4pTTlQvc6ocDKp0knQP7EiQuJas5MLlAatGVeV5cINdR",  help="API token")
    # parser.add_argument("--base-url", type=str, default="https://api.suanli.cn/v1/chat/completions", help="API base URL")
    # parser.add_argument("--model", type=str, default="deepseek-r1", help="Model name")

    # kimi
    parser.add_argument("--api-token", type=str, default="sk-HvOMXGI6kVzPvXlz7io4tUhvXG6dNPwQcQyRdrvYRtYvt5hL",  help="API token")
    parser.add_argument("--base-url", type=str, default="https://api.moonshot.cn/v1", help="API base URL")
    parser.add_argument("--model", type=str, default="moonshot-v1-128k", help="Model name")

    args = parser.parse_args()

    if args.method == 0:
        print("---------------   Method API!   ---------------")
        processor = SyncApiProcessor(
            api_token=args.api_token,
            base_url=args.base_url,
            model=args.model,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )

        processor.run(args.input_file, args.output_file, args.number)

    elif args.method == 1:
        # TODO()
        pass

    elif args.method == 2:
        # TODO()
        pass

    elif args.method == 3:
        # TODO()
        pass

    else:
        raise ValueError("Invalid method selected")


if __name__ == "__main__":
    main()
