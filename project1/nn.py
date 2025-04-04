import numpy as np
from typing import List, Dict, Any
import torch
import torch.nn as nn
from collections import Counter
import random
import os

# =================== 模型结构 ===================
class CharLevelCorrector(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, n_heads=4, num_layers=4, ff_dim=512, dropout=0.1, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, emb_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.size()
        emb = self.embedding(x) + self.pos_embedding[:, :seq_len, :]
        out = self.encoder(emb)
        return self.fc(out)


class NNCorrector:
    def __init__(self, model_path='char_corrector.pt'):
        self.vocab = None
        self.id2char = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path

        # 混淆集
        self.confusion_set = {
            '我': ['哦', '俄'], '是': ['事', '时'], '一': ['以', '已'],
            '国': ['果', '囯'], '人': ['入', '八'], '大': ['太', '犬'],
            '有': ['右', '又'], '在': ['再', '载'], '地': ['的', '底'],
            '在': ['再'], '的': ['地', '得']
        }

    def build_vocab(self, train_data, min_freq=1):
        counter = Counter()
        for sample in train_data:
            counter.update(sample['source'])
            counter.update(sample['target'])

        vocab = {'[PAD]': 0, '[UNK]': 1}
        for char, freq in counter.items():
            if freq >= min_freq:
                vocab[char] = len(vocab)
        return vocab

    def encode(self, text: str) -> List[int]:
        return [self.vocab.get(c, self.vocab['[UNK]']) for c in text]

    def decode(self, ids: List[int]) -> str:
        return ''.join([self.id2char[i] for i in ids if i in self.id2char and self.id2char[i] != '[PAD]'])

    def augment_sample(self, sample, prob=0.3):
        source = list(sample['source'])
        for i, char in enumerate(source):
            if char in self.confusion_set and random.random() < prob:
                source[i] = random.choice(self.confusion_set[char])
        return {'source': ''.join(source), 'target': sample['source']}

    def save_model(self):
        torch.save({
            'model_state': self.model.state_dict(),
            'vocab': self.vocab,
        }, self.model_path)

    def load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.vocab = checkpoint['vocab']
        self.id2char = {i: ch for ch, i in self.vocab.items()}
        self.model = CharLevelCorrector(len(self.vocab)).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

    def train(self, train_data: List[Dict[str, Any]], epochs=100, augment=True):
        if os.path.exists(self.model_path):
            print(f"Model already exists at '{self.model_path}', loading...")
            self.load_model()
            return

        # 构建词表和模型
        self.vocab = self.build_vocab(train_data)
        self.id2char = {i: ch for ch, i in self.vocab.items()}
        self.model = CharLevelCorrector(len(self.vocab)).to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(train_data)
            for sample in train_data:
                if sample.get('label', 1) == 0:
                    continue
                if augment and random.random() < 0.5:
                    sample = self.augment_sample(sample)

                src = self.encode(sample['source'])
                tgt = self.encode(sample['target'])

                max_len = max(len(src), len(tgt))
                src += [0] * (max_len - len(src))
                tgt += [0] * (max_len - len(tgt))

                src_tensor = torch.tensor([src], dtype=torch.long).to(self.device)
                tgt_tensor = torch.tensor([tgt], dtype=torch.long).to(self.device)

                logits = self.model(src_tensor)
                loss_raw = self.criterion(logits.view(-1, logits.size(-1)), tgt_tensor.view(-1))

                weight_mask = (src_tensor != tgt_tensor).float().view(-1) * 1.0 + 1.0
                loss = (loss_raw * weight_mask).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        self.save_model()
        print('Training completed and model saved!')

    def correct(self, text: str) -> str:
        if self.model is None:
            if os.path.exists(self.model_path):
                self.load_model()
            else:
                raise ValueError("Model not trained yet and no saved model found.")
        self.model.eval()
        with torch.no_grad():
            input_ids = self.encode(text)
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            logits = self.model(input_tensor)
            pred_ids = torch.argmax(logits, dim=-1)[0].tolist()
            return self.decode(pred_ids)
