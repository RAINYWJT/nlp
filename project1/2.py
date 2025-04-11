from typing import List, Dict, Any
import json
from evaluation import evaluate_performance, print_detailed_metrics

import torch
import torch.nn as nn
from transformers import BertTokenizer
from typing import List, Dict, Any
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

# 示例训练数据
train_data = load_data('data/train.jsonl')

test_data = load_data('data/test.jsonl')


print(test_data)

import re
from collections import defaultdict
import difflib
# 统计高频差异import difflib
from collections import defaultdict

def analyze_error_contexts(train_data):
    # 初始化统计字典
    replace_contexts = defaultdict(lambda: defaultdict(int))
    delete_contexts = defaultdict(lambda: defaultdict(int))
    insert_contexts = defaultdict(lambda: defaultdict(int))
    
    # 上下文窗口大小
    window_size = 2
    
    for item in train_data:
        if item['label'] == 1:
            source = item['source']
            target = item['target']
            diff = difflib.ndiff(source, target)
            diff_changes = [d for d in diff if d[0] != ' ']
            
            i = 0
            while i < len(diff_changes):
                # 获取上下文
                source_pos = sum(1 for d in diff_changes[:i] if d.startswith('-') or d.startswith(' '))
                left_context = source[max(0, source_pos-window_size):source_pos]
                right_context = source[source_pos:source_pos+window_size]
                context = (left_context, right_context)
                
                if diff_changes[i].startswith('-') and i+1 < len(diff_changes) and diff_changes[i+1].startswith('+'):
                    # 替换操作
                    from_char = diff_changes[i][2:]
                    to_char = diff_changes[i+1][2:]
                    replace_contexts[(from_char, to_char)][context] += 1
                    i += 2
                elif diff_changes[i].startswith('-'):
                    # 删除操作
                    char = diff_changes[i][2:]
                    delete_contexts[char][context] += 1
                    i += 1
                elif diff_changes[i].startswith('+'):
                    # 插入操作
                    char = diff_changes[i][2:]
                    insert_contexts[char][context] += 1
                    i += 1
    
    # 为每个错误类型选择最常见的上下文
    rule_set = {}
    
    # 替换规则
    for (from_char, to_char), contexts in replace_contexts.items():
        most_common_context = max(contexts.items(), key=lambda x: x[1])[0]
        rule_set[f"replace_{from_char}_with_{to_char}"] = {
            "type": "replace",
            "from": from_char,
            "to": to_char,
            "left_context": most_common_context[0],
            "right_context": most_common_context[1],
            "count": sum(contexts.values())
        }
    
    # 删除规则
    for char, contexts in delete_contexts.items():
        most_common_context = max(contexts.items(), key=lambda x: x[1])[0]
        rule_set[f"delete_{char}"] = {
            "type": "delete",
            "char": char,
            "left_context": most_common_context[0],
            "right_context": most_common_context[1],
            "count": sum(contexts.values())
        }
    
    # 插入规则
    for char, contexts in insert_contexts.items():
        most_common_context = max(contexts.items(), key=lambda x: x[1])[0]
        rule_set[f"insert_{char}"] = {
            "type": "insert",
            "char": char,
            "left_context": most_common_context[0],
            "right_context": most_common_context[1],
            "count": sum(contexts.values())
        }
    
    return rule_set

# 使用示例
rule_dictionary = analyze_error_contexts(train_data)

# 保存规则集
import json
with open('error_correction_rules.json', 'w', encoding='utf-8') as f:
    json.dump(rule_dictionary, f, ensure_ascii=False, indent=2)
# corrector = NNCorrector(train_data=train_data)
# # corrector = NNCorrector()
# corrector.train_model(train_data)
# # 纠正文本

# predictions = []
# for sample in tqdm(test_data[1:100]):
#     source = sample['source']
#     corrected = corrector.correct(source)
#     print(corrected)
#     predictions.append(
#         {'source': source, 'prediction': corrected, 'target': sample['target'], 'label': sample['label']}
#     )

#     # Calculate evaluation metrics
# metrics = evaluate_performance(predictions)
# print_detailed_metrics(metrics)