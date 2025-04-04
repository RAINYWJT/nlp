from nn import NNCorrector
import os
import json
import argparse
from typing import Dict, List, Tuple, Any
from tqdm import tqdm


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from jsonl file.

    Args:
        file_path: Path to the jsonl file.

    Returns:
        List of dictionaries containing the data.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

train_data = load_data('data/train.jsonl')
test_data = load_data('data/test.jsonl')

corrector = NNCorrector(model_path='char_corrector.pt')
corrector.train(train_data)

count = 0
for sample in test_data:

    source = sample['source']
    corrected = corrector.correct(source)
    print(sample['label'])
    print('t', sample['target'])
    print('c', corrected)
    print(' ')
    if count == 10:
        break
    count += 1