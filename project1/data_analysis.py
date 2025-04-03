#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data analysis module for Chinese Text Correction task.
This module provides functions for analyzing error patterns in the dataset.
"""

import re
import json
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict

# Try to import optional dependencies for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # 找到系统已有的中文字体 我是mac（这个是mac的官方指定字体位置）
    # TODO 如果想要使用字体，需要更换自己电脑的路径
    try:
        zh_font = "/System/Library/Fonts/Supplemental/Songti.ttc"  
        font_prop = fm.FontProperties(fname=zh_font)
        plt.rcParams["font.family"] = font_prop.get_name()
        plt.rcParams["axes.unicode_minus"] = False
    except:
        print("Warning: No suitable Chinese font found on this system. You may need to install a suitable font manually.")

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualization features will be disabled.")


def analyze_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the dataset to extract statistics and error patterns.

    Args:
        data: List of dictionaries containing the data.

    Returns:
        Dictionary containing analysis results.
    """
    # TODO 完成数据分析，可以从数据中观察统计信息、错误模式和难度分布等，帮助后续的方法设计。
    # print(data)
    total_samples = len(data)
    error_count = 0
    char_error_freq = Counter()
    error_patterns = Counter()
    
    for sample in data:
        source = sample.get("source", "")
        target = sample.get("target", "")
        label = sample.get("label", 0)
        
        # 只看错的
        if label == 1:  
            error_count += 1
            
            if len(source) == len(target):  
                for i, (s_char, t_char) in enumerate(zip(source, target)):
                    if s_char != t_char:
                        char_error_freq[s_char] += 1
                        error_patterns[(s_char, t_char)] += 1
    
    top_errors = error_patterns.most_common(10)
    top_error_chars = char_error_freq.most_common(10)
    
    return {
        "total_samples": total_samples,
        "error_samples": error_count,
        "error_rate": error_count / total_samples if total_samples > 0 else 0,
        "top_errors": top_errors,
        "top_error_chars": top_error_chars,
    }

def visualize_error_distribution(analysis_results: Dict[str, Any]) -> None:
    """
    Visualize the error distribution from analysis results.

    Args:
        analysis_results: Dictionary containing analysis results.
    """
    # TODO 可视化数据分析
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot visualize results: matplotlib not available.")
        return

    # 提取错误数据
    top_errors = analysis_results.get("top_errors", [])
    top_error_chars = analysis_results.get("top_error_chars", [])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 画出错误类型的分布（前 10）
    if top_errors:
        error_types = [f"{src}->{tgt}" for src, tgt in (pair for pair, _ in top_errors)]
        error_counts = [count for _, count in top_errors]

        axes[0].barh(error_types[::-1], error_counts[::-1], color="skyblue")
        axes[0].set_title("Top 10 Common Errors")
        axes[0].set_xlabel("Frequency")
        axes[0].set_ylabel("Error Type")

    # 画出最常见的错误字符（前 10）
    if top_error_chars:
        error_chars, char_counts = zip(*top_error_chars)
        axes[1].barh(error_chars[::-1], char_counts[::-1], color="salmon")
        axes[1].set_title("Top 10 Most Common Error Characters")
        axes[1].set_xlabel("Frequency")
        axes[1].set_ylabel("Character")

    plt.tight_layout()
    plt.savefig('./pic/error_distribution.png')
    plt.show()