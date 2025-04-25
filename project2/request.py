import json
import asyncio
import aiohttp
import argparse
import math
import logging
from tqdm import tqdm
from typing import List, Dict, Any

from utils import evaluate_response, Logger

logger = Logger().get_logger()


class ApiProcessor:
    def __init__(self, api_token: str, base_url: str, model: str, max_retries: int, retry_delay: int):
        if not api_token:
            raise ValueError("API token is required")
        self.api_token = api_token
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.headers = {"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"}
        self._processed_items = set()

    async def _process_item(self, item: Dict, session: aiohttp.ClientSession) -> Dict:
        """Process a single prompt asynchronously with retries."""
        max_delay = 32  # Maximum delay for exponential backoff
        current_delay = self.retry_delay

        # Handle potential nested structure if input comes from previous processing
        if 'response' in item and isinstance(item['response'], dict):
            item_data = item['response']
        else:
            item_data = item

        prompt = item_data.get("prompt", "")
        solution = item_data.get("solution", {})
        prompt_id = item_data.get("id", len(self._processed_items) + 1)
        self._processed_items.add(prompt_id)

        for attempt in range(self.max_retries):
            try:
                body = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "temperature": 0.7,
                }

                async with session.post(self.base_url, headers=self.headers, json=body) as response:
                    response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
                    response_data = await response.json()

                    if not response_data or 'choices' not in response_data or not response_data['choices']:
                        raise ValueError("Invalid response format from API")

                    response_text = response_data['choices'][0]['message']['content']
                    accuracy = evaluate_response(response_text, solution)
                    logging.info(f"Prompt #{prompt_id} - Accuracy: {accuracy}")
                    return {
                        "prompt": prompt,
                        "response": response_text,
                        "solution": solution,
                        "accuracy": accuracy,
                    }

            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError) as e:
                error_msg = f"Attempt {attempt + 1}/{self.max_retries} failed for Prompt #{prompt_id}..."
                if isinstance(e, aiohttp.ClientResponseError):
                    error_msg += f" HTTP Status: {e.status}, Message: {e.message}"
                elif isinstance(e, aiohttp.ClientConnectionError):
                    error_msg += f" Connection error: {str(e)}"
                elif isinstance(e, asyncio.TimeoutError):
                    error_msg += f" Timeout error: {str(e)}"
                elif isinstance(e, ValueError):
                    error_msg += f" Value error (likely invalid response): {str(e)}"
                elif isinstance(e, KeyError):
                    error_msg += f" Key error (likely missing data in response): {str(e)}"
                else:
                    error_msg += f" Error: {str(e)}"

                logging.warning(error_msg)  # Use warning for retriable errors

                if attempt == self.max_retries - 1:
                    logging.error(f"Failed to process prompt after {self.max_retries} attempts: #{prompt_id}...")
                    # Return error information instead of raising to allow processing other items
                    return {
                        "prompt": prompt,
                        "error": error_msg,
                        "solution": solution,
                        "accuracy": 0.0,  # Mark as inaccurate on failure
                    }

                # Exponential backoff
                await asyncio.sleep(current_delay)
                current_delay = min(current_delay * 2, max_delay)
            except Exception as e:
                # Catch unexpected errors
                logging.exception(f"Unexpected error processing Prompt #{prompt_id}... Error: {str(e)}")
                return {
                    "prompt": prompt,
                    "error": f"Unexpected error: {str(e)}",
                    "solution": solution,
                    "accuracy": 0.0,
                }
        # Should not be reached if max_retries > 0, but added for safety
        return {
            "prompt": prompt,
            "error": "Max retries exceeded without success or specific error.",
            "solution": solution,
            "accuracy": 0.0,
        }

    async def _process_batch(
        self, batch: List[Dict], semaphore: asyncio.Semaphore, session: aiohttp.ClientSession, pbar: tqdm
    ) -> List[Dict]:
        """Process a batch of prompts concurrently."""
        async with semaphore:
            tasks = [self._process_item(item, session) for item in batch]
            results = await asyncio.gather(*tasks)  # No return_exceptions=True, handle errors in _process_item
            for result in results:
                if "error" not in result:
                    pbar.update(1)  # Only update progress for successful items
                else:
                    # Log errors reported from _process_item
                    logging.error(
                        f"Item failed: {result.get('prompt', 'Unknown Prompt')[:50]}... Error: {result.get('error')}"
                    )
            return results

    async def run(self, input_file: str, output_file: str, batch_size: int, batch_group_size: int, max_concurrent: int):
        """Load data, process it in batches, and save results."""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
        except FileNotFoundError:
            logging.error(f"Input file not found: {input_file}")
            return
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from file: {input_file}")
            return

        # Adjust data structure if necessary (e.g., if nested under 'results')
        if isinstance(test_data, dict) and 'results' in test_data:
            items_to_process = test_data['results']
        elif isinstance(test_data, list):
            items_to_process = test_data
        else:
            logging.error("Unsupported input JSON structure.")
            return

        if not isinstance(items_to_process, list):
            logging.error("Input data 'results' key does not contain a list.")
            return

        if batch_size > 0:
            items_to_process = items_to_process[:batch_size]

        total_items = len(items_to_process)
        if total_items == 0:
            logging.info("No items to process.")
            return

        logging.info(f"Processing {total_items} prompts...")
        results = []
        total_accuracy = 0
        processed_count = 0

        semaphore = asyncio.Semaphore(max_concurrent)
        num_batches = math.ceil(total_items / batch_group_size)
        batches = [items_to_process[i * batch_group_size : (i + 1) * batch_group_size] for i in range(num_batches)]

        async with aiohttp.ClientSession() as session:
            with tqdm(total=total_items, desc="Processing prompts") as pbar:
                batch_results_list = await asyncio.gather(
                    *[self._process_batch(batch, semaphore, session, pbar) for batch in batches]
                )

            # Flatten results and calculate statistics
            for batch_result in batch_results_list:
                for item_result in batch_result:
                    results.append(item_result)
                    if "error" not in item_result:
                        processed_count += 1
                        total_accuracy += item_result.get("accuracy", 0.0)
                        #print(total_accuracy)

        avg_accuracy = total_accuracy / processed_count if processed_count > 0 else 0
        final_output = {
            "results": results,
            "statistics": {
                "total_samples_in_file": (
                    len(test_data['results'])
                    if isinstance(test_data, dict) and 'results' in test_data
                    else len(test_data) if isinstance(test_data, list) else 'N/A'
                ),
                "total_samples_attempted": total_items,
                "processed_samples_successfully": processed_count,
                "average_accuracy_on_success": avg_accuracy,
            },
        }

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, ensure_ascii=False, indent=2)
            logging.info(f"Successfully wrote results to {output_file}")
        except IOError as e:
            logging.error(f"Failed to write output file {output_file}: {e}")

        print(f"\nSuccessfully processed {processed_count}/{total_items} prompts")
        print(f"Average accuracy (on successfully processed): {avg_accuracy:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Process prompts using an API with object-oriented design.")

    parser.add_argument("-n", "--number", type=int, default=0, help="Number of prompts to process (0 for all)")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of prompts per batch group")
    parser.add_argument("--concurrent", type=int, default=5, help="Max concurrent requests")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retry times for each request")
    parser.add_argument("--retry-delay", type=int, default=1, help="Initial delay between retries in seconds")

    parser.add_argument(
        "--api-token",
        type=str,
        default="sk-HvOMXGI6kVzPvXlz7io4tUhvXG6dNPwQcQyRdrvYRtYvt5hL",
        help="API token",
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default="https://api.moonshot.cn/v1",
        help="API base URL",
    )

    parser.add_argument("--input-file", type=str, default='data/tc_200_zh.json', help="Input JSON file path")
    parser.add_argument("--output-file", type=str, default='output/tc_200_zh_output.json', help="Output JSON file path")

    parser.add_argument(
        "--model",
        type=str,
        default="moonshot-v1-32k", 
        help="Model",
    )

    args = parser.parse_args()


    processor = ApiProcessor(
        api_token=args.api_token,
        base_url=args.base_url,
        model=args.model,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )

    asyncio.run(
        processor.run(
            input_file=args.input_file,
            output_file=args.output_file,
            batch_size=args.number,  # Use 'number' for total limit
            batch_group_size=args.batch_size,  # Use 'batch_size' for concurrency groups
            max_concurrent=args.concurrent,
        )
    )


if __name__ == "__main__":
    main()
