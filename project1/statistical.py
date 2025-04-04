#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Statistical corrector for Chinese Text Correction task.
This module implements statistical methods for correcting errors in Chinese text.
"""
import random
import re
import json
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict
from difflib import SequenceMatcher

# Try to import optional dependencies
try:
    import jieba

    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("Warning: jieba not available. Some features will be disabled.")

    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    import numpy as np


    # Import CRF if available
    try:
        import sklearn_crfsuite
        from sklearn_crfsuite import metrics

        CRF_AVAILABLE = True
    except ImportError:
        CRF_AVAILABLE = False
        print("Warning: sklearn_crfsuite not available. CRF features will be disabled.")

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    CRF_AVAILABLE = False
    print("Warning: scikit-learn not available. Some features will be disabled.")

class StatisticalNgramCorrector:
    """
    A statistical corrector for Chinese text.
    """

    def __init__(self, lambda_1=0.1, lambda_2=0.3, lambda_3=0.4, lambda_4=0.2):
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()
        self.fourgram_counts = Counter()

        self.confusion_matrix = defaultdict(Counter)
        self.error_probs = defaultdict(float)
        self.phonetic_similarity = defaultdict(dict)
        self.visual_similarity = defaultdict(dict)

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4

        self.char_corrections = defaultdict(Counter)

    def train(self, train_data: List[Dict[str, Any]]) -> None:
        for sample in train_data:
            text = sample['target']
            for i in range(len(text)):
                self.unigram_counts[text[i]] += 1
                if i >= 1:
                    self.bigram_counts[text[i - 1] + text[i]] += 1
                if i >= 2:
                    self.trigram_counts[text[i - 2] + text[i - 1] + text[i]] += 1
                if i >= 3:
                    self.fourgram_counts[text[i - 3] + text[i - 2] + text[i - 1] + text[i]] += 1

            if sample['label'] == 1:
                source = sample['source']
                target = sample['target']
                if len(source) == len(target):
                    for i, (s_char, t_char) in enumerate(zip(source, target)):
                        if s_char != t_char:
                            left_context = source[max(0, i - 2): i]
                            right_context = source[i + 1: min(len(source), i + 3)]
                            context = left_context + '_' + right_context
                            self.confusion_matrix[(s_char, context)][t_char] += 1
                            self.confusion_matrix[(s_char, '')][t_char] += 1
                            self.error_probs[s_char] += 1
                            self.char_corrections[s_char][t_char] += 1

        for char, count in self.error_probs.items():
            self.error_probs[char] = count / self.unigram_counts.get(char, 1)

        print(
            f"Trained n-gram model with {len(self.unigram_counts)} unigrams, "
            f"{len(self.bigram_counts)} bigrams, {len(self.trigram_counts)} trigrams, "
            f"and {len(self.fourgram_counts)} 4-grams."
        )

    def correct(self, text: str) -> str:
        corrected_text = list(text)

        for i in range(len(text)):
            char = text[i]
            if self.error_probs.get(char, 0) < 0.01:
                continue

            left_context = text[max(0, i - 2): i]
            right_context = text[i + 1: min(len(text), i + 3)]
            context = left_context + '_' + right_context

            if (char, context) in self.confusion_matrix:
                correction = self.confusion_matrix[(char, context)].most_common(1)[0][0]
                corrected_text[i] = correction
                continue

            if (char, '') in self.confusion_matrix:
                correction = self.confusion_matrix[(char, '')].most_common(1)[0][0]
                if self.confusion_matrix[(char, '')][correction] > 2:
                    corrected_text[i] = correction
                    continue

            if self.error_probs.get(char, 0) >= 0.05 and i > 0 and i < len(text) - 1:
                candidates = set(list(self.unigram_counts.keys())[:500])
                for context_key in self.confusion_matrix:
                    if context_key[0] == char:
                        candidates.update(self.confusion_matrix[context_key].keys())

                best_score = -float('inf')
                best_char = char

                for candidate in candidates:
                    if candidate == char:
                        continue

                    score = 0
                    total_unigrams = sum(self.unigram_counts.values()) + len(self.unigram_counts)

                    unigram_prob = (self.unigram_counts.get(candidate, 0) + 1) / total_unigrams
                    score += self.lambda_1 * unigram_prob

                    if i >= 1:
                        bigram = text[i - 1] + candidate
                        bigram_prob = (self.bigram_counts.get(bigram, 0) + 1) / (
                            self.unigram_counts.get(text[i - 1], 0) + len(self.unigram_counts)
                        )
                        score += self.lambda_2 * bigram_prob

                    if i >= 2:
                        trigram = text[i - 2] + text[i - 1] + candidate
                        trigram_prob = (self.trigram_counts.get(trigram, 0) + 1) / (
                            self.bigram_counts.get(text[i - 2] + text[i - 1], 0) + len(self.unigram_counts)
                        )
                        score += self.lambda_3 * trigram_prob

                    if i >= 3:
                        fourgram = text[i - 3] + text[i - 2] + text[i - 1] + candidate
                        fourgram_prob = (self.fourgram_counts.get(fourgram, 0) + 1) / (
                            self.trigram_counts.get(text[i - 3] + text[i - 2] + text[i - 1], 0) + len(self.unigram_counts)
                        )
                        score += self.lambda_4 * fourgram_prob

                    if score > best_score:
                        best_score = score
                        best_char = candidate

                # Original character score
                original_score = 0
                original_unigram_prob = (self.unigram_counts.get(char, 0) + 1) / total_unigrams
                original_score += self.lambda_1 * original_unigram_prob

                if i >= 1:
                    bigram = text[i - 1] + char
                    bigram_prob = (self.bigram_counts.get(bigram, 0) + 1) / (
                        self.unigram_counts.get(text[i - 1], 0) + len(self.unigram_counts)
                    )
                    original_score += self.lambda_2 * bigram_prob

                if i >= 2:
                    trigram = text[i - 2] + text[i - 1] + char
                    trigram_prob = (self.trigram_counts.get(trigram, 0) + 1) / (
                        self.bigram_counts.get(text[i - 2] + text[i - 1], 0) + len(self.unigram_counts)
                    )
                    original_score += self.lambda_3 * trigram_prob

                if i >= 3:
                    fourgram = text[i - 3] + text[i - 2] + text[i - 1] + char
                    fourgram_prob = (self.fourgram_counts.get(fourgram, 0) + 1) / (
                        self.trigram_counts.get(text[i - 3] + text[i - 2] + text[i - 1], 0) + len(self.unigram_counts)
                    )
                    original_score += self.lambda_4 * fourgram_prob

                threshold = 1.2 + self.error_probs.get(char, 0) * 3
                if best_score > original_score * threshold:
                    corrected_text[i] = best_char

        return ''.join(corrected_text)

    
###################################################################################################################################################################################################



class StatisticalMLCorrector():
    def __init__(self):
         # Machine learning models
        self.error_detection_model = None  # 错误检测模型
        self.correction_model = None  # 纠错模型
        self.vectorizer = TfidfVectorizer(tokenizer=self._jieba_tokenize)  # TF-IDF 词向量
    
    def train(self, train_data: List[Dict[str, Any]]) -> None:
        """
        Train a machine learning model for text correction.

        Args:
            train_data: List of dictionaries containing the training data.
        """

        if not SKLEARN_AVAILABLE:
            print("Cannot train ML model: scikit-learn not available.")
            return

        # TODO 完成ml方法实现，可选择不同的文本编码方式、不同的特征提取和不同的模型, 推荐先使用一个模型检测，再使用一个模型来纠错。
        # 可以先将训练数据分为训练集和验证集，分别检测两个模型的效果，并调参
        # 可以使用数据增强或者预训练的词向量来提高模型的准确性
        return 
    
    
    def correct(self, text: str) -> str:
        """
        Correct text using machine learning model.

        Args:
            text: Input text.

        Returns:
            Corrected text.
        """
        return text