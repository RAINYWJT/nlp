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
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

class StatisticalMLCorrector:
    def __init__(self, confusion_dict=None, window_size=2):
        self.vectorizer = TfidfVectorizer()
        self.detection_model = SVC(kernel='linear', probability=True)
        self.correction_model = RandomForestClassifier()
        self.confusion_dict = confusion_dict if confusion_dict else {}
        self.window_size = window_size

    def extract_windows(self, text, window_size=2):
        """从文本中提取字符级上下文窗口片段"""
        padded = ["<PAD>"] * window_size + list(text) + ["<PAD>"] * window_size
        windows = []
        for i in range(window_size, len(padded) - window_size):
            context = padded[i - window_size:i + window_size + 1]
            windows.append("".join(context))
        return windows

    def generate_typo(self, text):
        """简单模拟错字生成"""
        chars = list(text)
        for i, c in enumerate(chars):
            if c in self.confusion_dict and np.random.rand() < 0.3:
                chars[i] = np.random.choice(self.confusion_dict[c])
        return "".join(chars)

    def prepare_training_data(self, samples):
        detection_texts = []
        detection_labels = []
        correction_texts = []
        correction_labels = []

        for sample in tqdm(samples):
            src = sample["source"]
            tgt = sample["target"]

            if len(src) != len(tgt):
                # 暂不支持对齐策略，跳过不等长样本
                continue

            padded_src = ["<"] * self.window_size + list(src) + [">"] * self.window_size
            padded_tgt = ["<"] * self.window_size + list(tgt) + [">"] * self.window_size

            for i in range(self.window_size, len(padded_src) - self.window_size):
                window = padded_src[i - self.window_size: i + self.window_size + 1]
                window_str = "".join(window)

                detection_texts.append(window_str)
                detection_labels.append(int(padded_src[i] != padded_tgt[i]))

                if padded_src[i] != padded_tgt[i]:
                    correction_texts.append(window_str)
                    correction_labels.append(padded_tgt[i])

        return detection_texts, detection_labels, correction_texts, correction_labels

    def train(self, samples):
        # 数据增强
        augmented = []
        for s in samples:
            noisy = self.generate_typo(s["source"])
            if noisy != s["target"]:
                augmented.append({"source": noisy, "target": s["target"]})
        samples += augmented

        # 准备训练数据
        print('prepare training data...')
        d_texts, d_labels, c_texts, c_labels = self.prepare_training_data(samples)

        # 训练检测器
        X_d = self.vectorizer.fit_transform(d_texts)
        y_d = np.array(d_labels)
        X_train_d, X_val_d, y_train_d, y_val_d = train_test_split(X_d, y_d, test_size=0.2, random_state=42)
        
        print(X_train_d, y_train_d)
        assert 0
        print('fit')
        self.detection_model.fit(X_train_d, y_train_d)
        print("Error Detection Report:\n", classification_report(y_val_d, self.detection_model.predict(X_val_d)))

        # 训练纠正器
        if c_texts:
            X_c = self.vectorizer.transform(c_texts)
            y_c = np.array(c_labels)
            self.correction_model.fit(X_c, y_c)
        else:
            print("No correction samples to train.")

    def correct(self, text):
        chars = list(text)
        windows = self.extract_windows(text, self.window_size)
        X = self.vectorizer.transform(windows)
        detection_preds = self.detection_model.predict(X)

        for i, is_wrong in enumerate(detection_preds):
            if is_wrong == 1:
                window = windows[i]
                pred_char = self.correction_model.predict(self.vectorizer.transform([window]))[0]
                chars[i] = pred_char
        return "".join(chars)
