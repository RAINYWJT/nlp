from nn import NNCorrector

corrector = NNCorrector()

import random
from nltk.corpus import wordnet
import nltk
import json
from typing import Dict, List, Any

# 下载 WordNet 数据
nltk.download('wordnet')
nltk.download('omw-1.4')

# 训练
train_data = [
    {"source": "敬请关注。", "target": "敬请关注。", "label": 0},
    {"source": "经请关注。", "target": "敬请关注。", "label": 1},
    # 更多训练样本...
]

def synonym_augmentation(text: str) -> str:
    """通过同义词替换增强文本"""
    words = text.split()
    new_words = []

    for word in words:
        # 查找该单词的同义词
        synonyms = wordnet.synsets(word)
        if synonyms:
            # 从同义词中随机选一个进行替换
            synonym = random.choice(synonyms).lemmas()[0].name()
            new_word = synonym if synonym != word else word
            new_words.append(new_word)
        else:
            new_words.append(word)
    
    return ' '.join(new_words)

def augment_data(train_data: list, augment_factor: int = 5) -> list:
    """通过数据增强生成更多样本"""
    augmented_data = []
    
    # 原始数据直接加入
    augmented_data.extend(train_data)

    for _ in range(augment_factor - 1):
        for sample in train_data:
            # 增强文本
            augmented_sample = sample.copy()
            augmented_sample['source'] = synonym_augmentation(sample['source'])
            augmented_data.append(augmented_sample)
    
    return augmented_data



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


# 示例训练数据
train_data = load_data('data/train.jsonl')


# # 使用数据增强生成100个样本
# augmented_data = augment_data(train_data, augment_factor=50)  # 这里设置增强5倍，变成100个样本

# # 打印增强后的数据
# for sample in augmented_data[:10]:  # 打印前10个样本
#     print(sample)


corrector.train_model(train_data[:50])
test_data = [
    {"source": "敬请管注。", "target": "敬请关注。", "label": 1},
    {"source": "尽情关注", "target": "敬请关注。", "label": 1},
    {"source": "尽情灌注", "target": "敬请关注。", "label": 1},
    {"source": "敬请关住。", "target": "敬请关注。", "label": 1},
]
# 纠正文本

for sample in test_data:
    corrected_text = corrector.correct(sample['source'])
    print(f"原始文本: {sample['target']}")
    print(f"纠正文本: {corrected_text}")
    print(' ')