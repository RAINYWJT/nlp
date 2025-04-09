from typing import List, Dict, Any
import json
from evaluation import evaluate_performance, print_detailed_metrics

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM
from typing import List, Dict, Any
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import BertTokenizer
from typing import List, Dict, Any
import random

class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, hidden_dropout_prob):
        super(BertEmbedding, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)  # For token type IDs (segment embeddings)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(input_ids.size(0), seq_length)
        token_type_ids = torch.zeros_like(input_ids, device=input_ids.device)

        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(BertSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        attention_output, _ = self.attention(hidden_states, hidden_states, hidden_states)
        attention_output = self.dense(attention_output)
        attention_output = self.dropout(attention_output)
        attention_output = self.LayerNorm(attention_output + hidden_states)  # Residual connection
        return attention_output


class BertLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(BertLayer, self).__init__()
        self.attention = BertSelfAttention(hidden_size, num_attention_heads)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        hidden_states = self.dense(attention_output)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + attention_output)  # Residual connection
        return hidden_states


# class BertForMaskedLM(nn.Module):
#     def __init__(self, vocab_size, hidden_size=768, num_attention_heads=12, num_layers=12, max_position_embeddings=512):
#         super(BertForMaskedLM, self).__init__()
#         self.embedding = BertEmbedding(vocab_size, hidden_size, max_position_embeddings, hidden_dropout_prob=0.1)
#         self.encoder_layers = nn.ModuleList([BertLayer(hidden_size, num_attention_heads) for _ in range(num_layers)])
#         self.dense = nn.Linear(hidden_size, vocab_size)
#         self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

#     def forward(self, input_ids, labels=None):
#         # Embedding Layer
#         embeddings = self.embedding(input_ids)

#         # Encoder
#         hidden_states = embeddings
#         for layer in self.encoder_layers:
#             hidden_states = layer(hidden_states)

#         # Output Layer
#         logits = self.dense(hidden_states)
        
#         # Calculate loss if labels are provided
#         loss = None
#         if labels is not None:
#             loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
#             loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
#         return (loss, logits) if loss is not None else logits


# class NNCorrector:
#     def __init__(self, model_name='bert-base-chinese', device='cpu'):
#         self.device = device
#         self.tokenizer = BertTokenizer.from_pretrained(model_name)
#         self.encoder = BertForMaskedLM(vocab_size=self.tokenizer.vocab_size).to(self.device)
#         self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-5)
#         self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

#     def augment_data(self, train_data: List[Dict[str, Any]], augment_prob: float = 0.9) -> List[Dict[str, Any]]:
#         augmented_data = []
        
#         for sample in train_data:
#             # 原始样本
#             source_text = sample['source']
#             target_text = sample['target']
#             label = sample['label']
            
#             # 数据增强：如果概率满足条件，交换 source 和 target
#             if random.random() < augment_prob:
#                 augmented_data.append({
#                     'source': target_text,   # 交换 source 和 target
#                     'target': source_text,   # 交换 source 和 target
#                     'label': 1 if label == 0 else 0  # 标签也要交换
#                 })
            
#             # 保留原始数据
#             augmented_data.append(sample)
        
#         return augmented_data

#     def train_model(self, train_data: List[Dict[str, Any]], augment_prob: float = 0.9) -> None:
#         self.encoder.train()
        
#         # 数据增强
#         augmented_train_data = self.augment_data(train_data, augment_prob)
#         print(len(augmented_train_data))
        
#         for batch in tqdm(augmented_train_data):
#             # Tokenize the source and target text
#             inputs = self.tokenizer(batch['source'], padding=True, truncation=True, return_tensors='pt').input_ids.to(self.device)
#             targets = self.tokenizer(batch['target'], padding=True, truncation=True, return_tensors='pt').input_ids.to(self.device)

#             if inputs.size(1) != targets.size(1):  # Compare sequence length, not batch size
#                 # Make sure inputs and targets have the same length (pad shorter one)
#                 max_length = max(inputs.size(1), targets.size(1))
#                 inputs = nn.functional.pad(inputs, (0, max_length - inputs.size(1)))
#                 targets = nn.functional.pad(targets, (0, max_length - targets.size(1)))

#             assert inputs.size(0) == targets.size(0)
            
#             self.optimizer.zero_grad()
            
#             # Forward pass
#             outputs = self.encoder(input_ids=inputs, labels=targets)
#             loss = outputs[0]
#             loss.backward()
#             self.optimizer.step()

#     def correct(self, text: str) -> str:
#         self.encoder.eval()
#         # Tokenize the input text
#         inputs = self.tokenizer(text, return_tensors='pt').input_ids.to(self.device)
        
#         with torch.no_grad():
#             outputs = self.encoder(input_ids=inputs)
        
#         # print(outputs)
#         logits = outputs  # No need for [1] because the output is just logits now
#         predicted_ids = torch.argmax(logits, dim=-1)
#         corrected_text = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
#         corrected_text = corrected_text.replace(" ", "")
#         return corrected_text

import torch
import torch.nn as nn
from transformers import BertTokenizer
from typing import List, Dict, Any
import random
from tqdm import tqdm

class BertForMaskedLMAndNSP(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_attention_heads=12, num_layers=12, max_position_embeddings=512):
        super(BertForMaskedLMAndNSP, self).__init__()
        self.embedding = BertEmbedding(vocab_size, hidden_size, max_position_embeddings, hidden_dropout_prob=0.1)
        self.encoder_layers = nn.ModuleList([BertLayer(hidden_size, num_attention_heads) for _ in range(num_layers)])
        self.dense = nn.Linear(hidden_size, vocab_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        
        # NSP classifier
        self.nsp_dense = nn.Linear(hidden_size, 2)  # 2 labels for NSP: 0 or 1

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, next_sentence_labels=None):
        # Embedding Layer
        embeddings = self.embedding(input_ids)

        # Encoder
        hidden_states = embeddings
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states)

        # Output Layer for MLM
        mlm_logits = self.dense(hidden_states)
        
        # NSP (Next Sentence Prediction) Layer
        # We take the hidden state of the first token ([CLS]) for NSP prediction
        nsp_logits = self.nsp_dense(hidden_states[:, 0])  # Only [CLS] token
        
        # Calculate losses
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
            mlm_loss = loss_fn(mlm_logits.view(-1, mlm_logits.size(-1)), labels.view(-1))
            
            # NSP Loss
            if next_sentence_labels is not None:
                nsp_loss = loss_fn(nsp_logits.view(-1, nsp_logits.size(-1)), next_sentence_labels.view(-1))
                loss = mlm_loss + nsp_loss
            else:
                loss = mlm_loss
        
        return (loss, mlm_logits, nsp_logits) if loss is not None else (mlm_logits, nsp_logits)


class NNCorrector:
    def __init__(self, model_name='bert-base-chinese', device='cpu'):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.encoder = BertForMaskedLMAndNSP(vocab_size=self.tokenizer.vocab_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-5)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def augment_data(self, train_data: List[Dict[str, Any]], augment_prob: float = 0.9) -> List[Dict[str, Any]]:
        augmented_data = []
        
        for sample in train_data:
            source_text = sample['source']
            target_text = sample['target']
            label = sample['label']
            
            if random.random() < augment_prob:
                augmented_data.append({
                    'source': target_text,
                    'target': source_text,
                    'label': 1 if label == 0 else 0
                })
            
            augmented_data.append(sample)
        
        return augmented_data

    def train_model(self, train_data: List[Dict[str, Any]], augment_prob: float = 0.9) -> None:
        self.encoder.train()
        
        augmented_train_data = self.augment_data(train_data, augment_prob)
        print(len(augmented_train_data))
        
        for batch in tqdm(augmented_train_data):
            inputs = self.tokenizer(batch['source'], padding=True, truncation=True, return_tensors='pt').input_ids.to(self.device)
            targets = self.tokenizer(batch['target'], padding=True, truncation=True, return_tensors='pt').input_ids.to(self.device)
            
            if inputs.size(1) != targets.size(1):
                max_length = max(inputs.size(1), targets.size(1))
                inputs = nn.functional.pad(inputs, (0, max_length - inputs.size(1)))
                targets = nn.functional.pad(targets, (0, max_length - targets.size(1)))

            assert inputs.size(0) == targets.size(0)

            # Generate random next sentence labels (0 or 1)
            next_sentence_labels = torch.randint(0, 2, (inputs.size(0),), device=self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.encoder(input_ids=inputs, labels=targets, next_sentence_labels=next_sentence_labels)
            loss = outputs[0]
            loss.backward()
            self.optimizer.step()

    def correct(self, text: str) -> str:
        self.encoder.eval()
        inputs = self.tokenizer(text, return_tensors='pt').input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.encoder(input_ids=inputs)
        
        logits = outputs[0]
        predicted_ids = torch.argmax(logits, dim=-1)
        corrected_text = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        corrected_text = corrected_text.replace(" ", "")
        return corrected_text



# 初始化模型
corrector = NNCorrector()

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
# corrector = NNCorrector()
corrector.train_model(train_data)
# 纠正文本

predictions = []
for sample in tqdm(test_data[0:100]):
    source = sample['source']
    corrected = corrector.correct(source)
    print(corrected)
    predictions.append(
        {'source': source, 'prediction': corrected, 'target': sample['target'], 'label': sample['label']}
    )

    # Calculate evaluation metrics
metrics = evaluate_performance(predictions)
print_detailed_metrics(metrics)