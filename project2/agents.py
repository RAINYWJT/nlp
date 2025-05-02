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

    def run(self):
        pass