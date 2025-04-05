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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import jieba  # 中文分词

class StatisticalMLCorrector:
    def __init__(self, encoder_type='tfidf', detection_model='svm', correction_model='random_forest'):
        """
        Initialize the corrector with encoding type, detection model, and correction model.
        """
        self.encoder_type = encoder_type
        self.detection_model = detection_model
        self.correction_model = correction_model
        self.vectorizer = None
        self.detection_model_obj = None
        self.correction_model_obj = None

    def encode_text(self, texts):
        """Encode Chinese text using TF-IDF or other encoders."""
        if self.encoder_type == 'tfidf':
            # 中文分词
            texts = [" ".join(jieba.cut(text)) for text in texts]
            vectorizer = TfidfVectorizer()
            encoded_text = vectorizer.fit_transform(texts).toarray()
            return encoded_text, vectorizer
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")

    def data_augmentation(self, text):
        """Apply simple data augmentation like synonym replacement."""
        words = text.split()
        augmented_text = words.copy()
        for i in range(len(words)):
            # 这里可以扩展同义词替换的策略
            # 示例中没有同义词词库，假设可以用 WordNet 获取同义词（需要调整中文同义词库）
            synonyms = words[i]  # 这里没有实际同义词库，直接使用原词
            if synonyms != words[i]:
                augmented_text[i] = synonyms
        return ' '.join(augmented_text)

    def train(self, train_data):
        """Train error detection and correction models."""
        source = [sample['source'] for sample in train_data]
        target = [sample['target'] for sample in train_data]
        label = [sample['label'] for sample in train_data]

        # Data augmentation: augment the source and target data
        augmented_sources = [self.data_augmentation(text) for text in source]
        augmented_targets = [self.data_augmentation(text) for text in target]  # Augment target data as well
        source += augmented_sources
        target += augmented_targets  # Make sure target and source data are augmented equally
        label += label  # Keep the same labels for augmented data
        print(source, target, label)

        # Split into train and validation sets
        X, self.vectorizer = self.encode_text(source)
        label = np.array(label)
        print(X.shape, label.shape)
        X_train, X_val, y_train, y_val, target_train, target_val = train_test_split(X, label, target, test_size=0.3, random_state=42)

        print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
    

        # Train error detection model
        if self.detection_model == 'svm':
            self.detection_model_obj = SVC(kernel='rbf')
        elif self.detection_model == 'random_forest':
            self.detection_model_obj = RandomForestClassifier()
        self.detection_model_obj.fit(X_train, y_train)

        # Evaluate error detection model
        y_val_pred = self.detection_model_obj.predict(X_val)
        print(f"Error detection model accuracy: {accuracy_score(y_val, y_val_pred)}")

        # Train error correction model
        if self.correction_model == 'logistic_regression':
            self.correction_model_obj = LogisticRegression()
            self.correction_model_obj.fit(X_train, target_train)
        elif self.correction_model == 'random_forest':
            self.correction_model_obj = RandomForestClassifier()
            self.correction_model_obj.fit(X_train, target_train)
        else:
            raise ValueError(f"Unsupported correction model: {self.correction_model}")

    def correct(self, text):
        """Use the trained models to detect and correct errors."""
        # Step 1: Use error detection model
        X = self.vectorizer.transform([text]).toarray()
        detection_result = self.detection_model_obj.predict(X)

        # Step 2: If an error is detected, use the correction model
        if detection_result == 1:  # Error detected
            corrected_text = self.correction_model_obj.predict(X)[0]
            # print(f"Corrected text: {corrected_text}")
            return corrected_text
        else:
            return text  # No error detected, return original text
