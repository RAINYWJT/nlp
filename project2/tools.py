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

        self.client = OpenAI(
            api_key=self.api_token,
            base_url=self.base_url
        )

    def extract_clues(self, prompt: str) -> List[str]:
        """Extract key clues from the puzzle prompt using LLM"""
        system_msg = """You are a detective assistant that extracts key logical clues from murder mystery puzzles. 
        Extract ONLY the factual clues, ignoring narrative descriptions. List each clue on a new line."""
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                clues = response.choices[0].message.content.split('\n')
                return [clue.strip() for clue in clues if clue.strip()]
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_delay)
        
        return []

    def logical_solver(self, clues: List[str], question: str) -> Dict:
        """Solve the logic puzzle using extracted clues"""
        system_msg = """You are an expert logic puzzle solver. Use the given clues to solve the mystery.
        Respond with ONLY a JSON object containing the answers to all questions in the format:
        {"A": "answer1", "B": "answer2", ...}"""
        
        clues_text = "\n".join(f"- {clue}" for clue in clues)
        prompt = f"Clues:\n{clues_text}\n\nQuestion: {question}"
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_delay)
        
        return {}

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
                # Extract key clues
                clues = self.extract_clues(prompt)
                
                # Solve the puzzle
                result = self.logical_solver(clues, question)

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
                    "clues": clues,
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
