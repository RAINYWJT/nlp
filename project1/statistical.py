#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Statistical corrector for Chinese Text Correction task.
This module implements statistical methods for correcting errors in Chinese text.
"""

import random
from typing import Dict, List, Any
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
import random

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

        # TODO() 可以选择查看
        # print(
        #     f"Trained n-gram model with {len(self.unigram_counts)} unigrams, "
        #     f"{len(self.bigram_counts)} bigrams, {len(self.trigram_counts)} trigrams, "
        #     f"and {len(self.fourgram_counts)} 4-grams."
        # )

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

            if self.error_probs.get(char, 0) >= 0.3 and i > 0 and i < len(text) - 1:
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

# 上下文窗口提取器：将文本按字符滑动窗口方式提取局部上下文
class ContextWindowExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, window_size=3):
        self.window_size = window_size  # 滑动窗口大小

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        contexts = []
        for text in X:
            # 在首尾填充 [PAD] 以保证窗口完整
            padded = ['[PAD]'] * (self.window_size // 2) + list(text) + ['[PAD]'] * (self.window_size // 2)
            for i in range(len(text)):
                window = padded[i:i + self.window_size]
                contexts.append("".join(window))  # 将窗口内字符合并为字符串
        return contexts

# 统计机器学习纠错器：检测 + 替换字符级别错误
class StatisticalMLCorrector:
    def __init__(self, confusion_dict=None, window_size=3, seed = 1):
        self.confusion_dict = confusion_dict or {}
        self.window_size = window_size
        self.detector = None        # 错误检测器（分类器）
        self.corrector = None       # 错误纠正器（字符分类器）
        # 使用 TF-IDF 提取特征：字符 n-gram（1 到 6 元）
        self.detect_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(1, 6),
            max_features=30000,
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )
        self.correct_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(1, 6),
            max_features=30000,
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )
        self.seed = seed 
        random.seed(seed)

    def _augment_error_sample(self, src: str, tgt: str, n=3):
        augmented_list = []
        flags_list = []
        for _ in range(n):
            noisy_src = list(src)
            augmented = []
            ys = []
            for i in range(len(src)):
                if src[i] != tgt[i]:
                    augmented.append(noisy_src[i])
                    ys.append(1)
                else:
                    if noisy_src[i] in self.confusion_dict and random.random() < 0.1:
                        augmented.append(random.choice(self.confusion_dict[noisy_src[i]]))
                        ys.append(1)
                    else:
                        augmented.append(noisy_src[i])
                        ys.append(0)
            augmented_list.append("".join(augmented))
            flags_list.append(ys)
        return augmented_list, flags_list


    # 提取用于检测模型训练的样本（窗口 + 是否错误）
    def _extract_detection_samples(self, sources, targets, labels):
        contexts, ys = [], []
        for src, tgt, lbl in zip(sources, targets, labels):
            if not src or not tgt:
                continue  # 跳过空字符串

            if lbl == 1 and len(src) == len(tgt):
                # 原始错误样本处理
                padded = ['[PAD]'] * (self.window_size // 2) + list(src) + ['[PAD]'] * (self.window_size // 2)
                for i in range(len(src)):
                    window = padded[i:i + self.window_size]
                    contexts.append("".join(window))
                    ys.append(int(src[i] != tgt[i]))

                # 多个增强样本处理（增强数量你可以改，比如 3）
                augmented_list, flags_list = self._augment_error_sample(src, tgt)
                for aug_src, aug_flags in zip(augmented_list, flags_list):
                    padded_aug = ['[PAD]'] * (self.window_size // 2) + list(aug_src) + ['[PAD]'] * (self.window_size // 2)
                    for i in range(len(aug_src)):
                        window = padded_aug[i:i + self.window_size]
                        contexts.append("".join(window))
                        ys.append(aug_flags[i])
        return contexts, ys



    # 提取用于纠错模型训练的样本（错误窗口 + 正确字符标签）
    def _extract_correction_samples(self, sources, targets, labels):
        contexts, y_chars = [], []
        for src, tgt, lbl in zip(sources, targets, labels):
            if lbl == 1 and len(src) == len(tgt):
                padded = ['[PAD]'] * (self.window_size // 2) + list(src) + ['[PAD]'] * (self.window_size // 2)
                for i in range(len(src)):
                    if src[i] != tgt[i]:
                        window = padded[i:i + self.window_size]
                        contexts.append("".join(window))
                        y_chars.append(tgt[i])  # 正确的字符标签
        return contexts, y_chars

    # 模型训练入口
    def train(self, train_data: List[Dict[str, Any]]) -> None:
        sources = [item['source'] for item in train_data]  # 原始文本
        targets = [item['target'] for item in train_data]  # 正确文本
        labels = [item['label'] for item in train_data]    # 标签：是否有误

        # 1. 训练错误检测模型
        print('[检测模型训练]')
        detect_X, detect_y = self._extract_detection_samples(sources, targets, labels)
        detect_X_vec = self.detect_vectorizer.fit_transform(detect_X)
        self.detector = SGDClassifier(loss='perceptron', class_weight='balanced', max_iter=1000, n_jobs=-1, random_state=self.seed)
        self.detector.fit(detect_X_vec, detect_y)
        print(classification_report(detect_y, self.detector.predict(detect_X_vec)))

        # 2. 训练字符纠错模型
        print('[纠错模型训练]')
        corr_X, corr_y = self._extract_correction_samples(sources, targets, labels)
        corr_X_vec = self.correct_vectorizer.fit_transform(corr_X)
        # self.corrector = SGDClassifier(loss='huber', max_iter=1000, n_jobs=-1,random_state=self.seed)
        self.corrector = RandomForestClassifier(n_estimators=500)
        self.corrector.fit(corr_X_vec, corr_y)
        # print(classification_report(corr_y, self.corrector.predict(corr_X_vec)))

    # 文本纠错函数：检测 + 替换错误字符
    def correct(self, text: str) -> str:
        corrected = list(text)
        padded = ['[PAD]'] * (self.window_size // 2) + list(text) + ['[PAD]'] * (self.window_size // 2)
        windows = ["".join(padded[i:i + self.window_size]) for i in range(len(text))]

        # 批量提取特征并检测错误位置
        detect_vecs = self.detect_vectorizer.transform(windows)
        is_wrong = self.detector.predict(detect_vecs)

        if any(is_wrong):
            # 提取被检测为错误的窗口和位置
            to_correct_windows = [windows[i] for i, w in enumerate(is_wrong) if w]
            to_correct_indices = [i for i, w in enumerate(is_wrong) if w]
            corr_vecs = self.correct_vectorizer.transform(to_correct_windows)
            corr_preds = self.corrector.predict(corr_vecs)
            # 替换错误字符为预测结果
            for idx, pred_char in zip(to_correct_indices, corr_preds):
                corrected[idx] = pred_char
        return "".join(corrected)
