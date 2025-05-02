import json
import time
import argparse
from tqdm import tqdm
from typing import List, Dict
from openai import OpenAI
import re
from utils import *

from api_engineer import SyncApiProcessor
from prompt_engineer import PromptEngineer



def main():
    parser = argparse.ArgumentParser(description="Simplified API Processor (sync version)")
    parser.add_argument("--method",type=int, default=0, help="Number of prompts to process: (0: API),(1: Prompt Engineer),(2:Tools),(3:Agent)")


    parser.add_argument("-n", "--number", type=int, default=0, help="Number of prompts to process (0 for all)")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retry times")
    parser.add_argument("--retry-delay", type=int, default=1, help="Initial delay between retries (seconds)")
   
    parser.add_argument("--input-file", type=str, default='data/tc_200_zh.json', help="Input file")
    parser.add_argument("--output-file", type=str, default='output/tc_200_zh_output1.json', help="Output file")
    
    # free QWQ
    # parser.add_argument("--api-token", type=str, default="sk-hlYe4pTTlQvc6ocDKp0knQP7EiQuJas5MLlAatGVeV5cINdR",  help="API token")
    # parser.add_argument("--base-url", type=str, default="https://api.suanli.cn/v1/chat/completions", help="API base URL")
    # parser.add_argument("--model", type=str, default="deepseek-r1", help="Model name")

    # kimi
    parser.add_argument("--api-token", type=str, default="sk-HvOMXGI6kVzPvXlz7io4tUhvXG6dNPwQcQyRdrvYRtYvt5hL",  help="API token")
    parser.add_argument("--base-url", type=str, default="https://api.moonshot.cn/v1", help="API base URL")
    parser.add_argument("--model", type=str, default="moonshot-v1-32k", help="Model name")

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
        print("---------------   Method Prompt Engineer!   ---------------")
        processor = PromptEngineer(
            api_token=args.api_token,
            base_url=args.base_url,
            model=args.model,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )

        processor.run(args.input_file, args.output_file, args.number)

    elif args.method == 2:
        print("---------------   Method Tools!   ---------------")
        # TODO()
        pass

    elif args.method == 3:
        print("---------------   Method Agents!   ---------------")
        # TODO()
        pass

    else:
        raise ValueError("Invalid method selected")


if __name__ == "__main__":
    main()
