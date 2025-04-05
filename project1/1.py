import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import jieba  # 中文分词

class StatisticalMLCorrector:
    def __init__(self, encoder_type='tfidf', detection_model='svm', correction_model='logistic_regression'):
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

        print('predict', detection_result)
        # Step 2: If an error is detected, use the correction model
        if detection_result == 1:  # Error detected
            corrected_text = self.correction_model_obj.predict(X)[0]
            # print(f"Corrected text: {corrected_text}")
            return corrected_text
        else:
            return text  # No error detected, return original text


# Example usage:
from typing import Dict, List, Tuple, Any
import json
# Sample train data
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

# Initialize and train model
corrector = StatisticalMLCorrector(encoder_type='tfidf', detection_model='random_forest', correction_model='random_forest')
corrector.train(train_data)

# Test correction
text = [item['source'] for item in test_data[0:10]]
targ = [item['target'] for item in test_data[0:10]]
label = [item['label'] for item in test_data[0:10]]

corrected_texts = [corrector.correct(t) for t in text]
# 假设 text, targ 和 corrected_texts 都是列表
for original_text, target_text, corrected_text, test_label in zip(text, targ, corrected_texts, label):
    print('label', test_label)
    print(f"Original text: {original_text}")
    print(f"Target text: {target_text}")
    print(f"Corrected text: {corrected_text}")
    print("-" * 50)  # 分隔符，用于清晰显示不同文本的对比
