# class NNCorrector:
#     def __init__(self):
#         pass

#     def train(self, train_data: List[Dict[str, Any]]) -> None:
#         pass

#     def correct(self, text: str) -> str:
#         pass

# 由于任务较难，建议使用深度学习方法获得更好得分。

# 注意：

# 禁止使用LLM纠错
# 禁止使用现成的文本纠错工具和已经训练完成文本纠错模型
# 数据输入格式为：{"source": "敬请关注。", "target": "敬请关注。", "label": 0}
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any
from collections import Counter
import re

class NNCorrector(nn.Module):
    def __init__(self, max_seq_length=100, embedding_dim=128, hidden_dim=256):
        super(NNCorrector, self).__init__()
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self.label_threshold = 0.8
        
        # Initialize model components
        self.embedding = None
        self.encoder = None
        self.attention = None
        self.char_decoder = None
        self.classifier = None
        
    def build_vocabulary(self, texts: List[str]) -> None:
        chars = set()
        for text in texts:
            chars.update(text)
        
        self.char_to_idx = {'<PAD>': 0, '<UNK>': 1}
        for idx, char in enumerate(sorted(chars), start=2):
            self.char_to_idx[char] = idx
        
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        # Initialize model components after vocab is built
        self._init_model()
    
    def _init_model(self):
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=0
        )
        
        # Bidirectional LSTM encoder
        self.encoder = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim * 2,  # *2 for bidirectional
            num_heads=4,
            batch_first=True
        )
        
        # Character decoder (time distributed)
        self.char_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),  # *4 because we concat lstm and attention
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.vocab_size)
        )
        
        # Correction classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 * self.max_seq_length, 512),  # Flattened sequence
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def preprocess_data(self, data: List[Dict[str, Any]]) -> tuple:
        sources = [item['source'] for item in data]
        targets = [item['target'] for item in data]
        
        if not self.char_to_idx:
            self.build_vocabulary(sources + targets)
        
        source_seqs = self.texts_to_sequences(sources)
        target_seqs = self.texts_to_sequences(targets)
        
        labels = [int(item['source'] != item['target']) for item in data]
        
        return (
            torch.LongTensor(source_seqs),
            torch.LongTensor(target_seqs),
            torch.FloatTensor(labels).unsqueeze(1)
        )
    
    def texts_to_sequences(self, texts: List[str]) -> np.ndarray:
        sequences = []
        for text in texts:
            seq = [self.char_to_idx.get(c, self.char_to_idx['<UNK>']) 
                   for c in text[:self.max_seq_length]]
            seq += [self.char_to_idx['<PAD>']] * (self.max_seq_length - len(seq))
            sequences.append(seq)
        return np.array(sequences)
    
    def sequences_to_texts(self, sequences: np.ndarray) -> List[str]:
        texts = []
        for seq in sequences:
            text = ''.join([self.idx_to_char.get(idx, '') for idx in seq if idx > 0])
            texts.append(text)
        return texts
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM encoding
        lstm_out, _ = self.encoder(embedded)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Combine LSTM and attention
        combined = torch.cat([lstm_out, attn_out], dim=-1)
        
        # Character predictions
        char_preds = self.char_decoder(combined)
        
        # Correction classification
        batch_size = x.size(0)
        cls_input = lstm_out.reshape(batch_size, -1)  # 使用reshape替代view
        cls_pred = self.classifier(cls_input)
        
        return char_preds, cls_pred
    
    def train_model(self, train_data: List[Dict[str, Any]], epochs=10, batch_size=32, lr=0.001):
        #print(train_data)
        source_seqs, target_seqs, labels = self.preprocess_data(train_data)
        
        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(source_seqs, target_seqs, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion_char = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        criterion_cls = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        self.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                src, tgt, lbl = batch
                
                optimizer.zero_grad()
                
                # Forward pass
                char_preds, cls_pred = self(src)
                
                # Reshape for cross entropy (combine batch and seq dimensions)
                char_loss = criterion_char(
                    char_preds.view(-1, self.vocab_size),
                    tgt.view(-1)
                )
                
                cls_loss = criterion_cls(cls_pred, lbl)
                
                # Combined loss
                loss = 0.7 * char_loss + 0.3 * cls_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    def correct(self, text: str) -> str:
        self.eval()
        
        if not self.char_to_idx:
            return text
        
        # Prepare input
        seq = self.texts_to_sequences([text])
        seq_tensor = torch.LongTensor(seq)
        
        with torch.no_grad():
            char_preds, cls_pred = self(seq_tensor)
        
        # Check if correction is needed
        if cls_pred.item() < self.label_threshold:
            return text
        
        # Get predicted characters
        predicted_indices = torch.argmax(char_preds, dim=-1).squeeze(0).numpy()
        corrected_text = self.sequences_to_texts([predicted_indices])[0]
        
        return corrected_text if corrected_text != text else text