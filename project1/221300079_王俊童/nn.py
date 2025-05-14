import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM
from typing import List, Dict, Any
from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

class NNPreCorrector:
    def __init__(self, model_name='bert-base-chinese', device='cpu'):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.encoder = BertForMaskedLM.from_pretrained(model_name).to(self.device)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-5)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def train_model(self, train_data: List[Dict[str, Any]]) -> None:
        self.encoder.train()
        for batch in tqdm(train_data):
            # Tokenize the source and target text
            inputs = self.tokenizer(batch['source'], padding=True, truncation=True, return_tensors='pt').input_ids.to(self.device)
            targets = self.tokenizer(batch['target'], padding=True, truncation=True, return_tensors='pt').input_ids.to(self.device)

            if inputs.size(1) != targets.size(1):  # Compare sequence length, not batch size
                # Make sure inputs and targets have the same length (pad shorter one)
                max_length = max(inputs.size(1), targets.size(1))
                inputs = nn.functional.pad(inputs, (0, max_length - inputs.size(1)))
                targets = nn.functional.pad(targets, (0, max_length - targets.size(1)))

            # print(inputs.shape , targets.shape)
            assert inputs.size(0) == targets.size(0)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.encoder(input_ids=inputs, labels=targets)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()

    def correct(self, text: str) -> str:
        self.encoder.eval()
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors='pt').input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.encoder(input_ids=inputs)
        
        logits = outputs.logits
        predicted_ids = torch.argmax(logits, dim=-1)
        corrected_text = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        corrected_text = corrected_text.replace(" ", "")
        return corrected_text


############################################################################################################################################################################

class ImprovedCorrector(nn.Module):
    def __init__(self, char2idx, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        self.char2idx = char2idx
        self.idx2char = {v: k for k, v in char2idx.items()}
        self.vocab_size = len(char2idx)
        
        # 增强的模型架构
        self.embedding = nn.Embedding(self.vocab_size, 256)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1
            ),
            num_layers=4
        )
        self.decoder = nn.LSTM(256, 512, bidirectional=True, num_layers=2, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, self.vocab_size)
        )
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x.permute(1, 0, 2))  # Transformer需要序列维度在前
        x, _ = self.decoder(x.permute(1, 0, 2))
        return self.fc(x)

class CorrectorDataset(Dataset):
    def __init__(self, data, char2idx, noise_prob=0.3):
        self.data = data
        self.char2idx = char2idx
        self.noise_prob = noise_prob
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        clean = self.data[idx]['target']
        # 数据增强：动态生成带噪声的输入
        noisy = self.add_noise(clean)
        return self.text_to_ids(noisy), self.text_to_ids(clean)
    
    def add_noise(self, text):
        # 多种噪声类型：增删改
        chars = list(text)
        if random.random() < self.noise_prob and len(chars) > 1:
            noise_type = random.choice(['delete', 'replace', 'insert'])
            if noise_type == 'delete':
                del_idx = random.randint(0, len(chars)-1)
                del chars[del_idx]
            elif noise_type == 'replace':
                rep_idx = random.randint(0, len(chars)-1)
                chars[rep_idx] = random.choice(list(self.char2idx.keys()))
            elif noise_type == 'insert':
                ins_idx = random.randint(0, len(chars))
                chars.insert(ins_idx, random.choice(list(self.char2idx.keys())))
        return ''.join(chars)
    
    def text_to_ids(self, text):
        return [self.char2idx.get(c, self.char2idx['<unk>']) for c in text]

class NNCorrector:
    def __init__(self, train_data, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.char2idx = self.build_vocab(train_data)
        self.idx2char = {v: k for k, v in self.char2idx.items()}  # Add this line to create the reverse mapping
        self.model = ImprovedCorrector(self.char2idx, device).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=3)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.char2idx['<pad>'])
        self.vocab_size = len(self.char2idx)

    def build_vocab(self, data):
        char_counts = defaultdict(int)
        for item in data:
            for c in item['target']:
                char_counts[c] += 1
        chars = ['<pad>', '<unk>', '<sos>', '<eos>'] + [c for c in char_counts if char_counts[c] > 1]
        return {c: i for i, c in enumerate(chars)}
    
    def collate_fn(self, batch):
        inputs, targets = zip(*batch)
        max_len = max(max(len(x) for x in inputs), max(len(x) for x in targets))
        
        def pad_seq(seqs):
            padded = torch.full((len(seqs), max_len), self.char2idx['<pad>'], dtype=torch.long)
            for i, seq in enumerate(seqs):
                padded[i, :len(seq)] = torch.tensor(seq)
            return padded
        
        return pad_seq(inputs), pad_seq(targets)
    
    def train_model(self, train_data, epochs=1, batch_size=32):
        dataset = CorrectorDataset(train_data, self.char2idx)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        
        best_loss = float('inf')
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for inputs, targets in tqdm(loader, desc=f'Epoch {epoch+1}'):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                loss = self.loss_fn(outputs.view(-1, self.vocab_size), targets.view(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            self.scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
    
    def correct(self, text: str, noise_prob: float = 0.2) -> str:
        self.model.eval()
        with torch.no_grad():
            # 创建临时数据集实例用于添加噪声
            temp_dataset = CorrectorDataset([], self.char2idx, noise_prob=noise_prob)
            noisy = temp_dataset.add_noise(text)  # 正确的噪声生成方式
            
            input_ids = self.text_to_ids(noisy)
            input_tensor = torch.tensor([input_ids], device=self.device)
            
            outputs = self.model(input_tensor)
            pred_ids = outputs.argmax(-1).squeeze().cpu().tolist()
            
            corrected = self.ids_to_text(pred_ids)
            legal_chars = set(self.char2idx.keys()) - {'<unk>', '<pad>'}
            corrected = ''.join(c for c in corrected if c in legal_chars)
            return self.post_process(corrected, original=text)

    def text_to_ids(self, text):
        return [self.char2idx.get(c, self.char2idx['<unk>']) for c in text]
    
    def ids_to_text(self, ids):
        return ''.join([self.idx2char.get(i, '<unk>') for i in ids if i not in {self.char2idx['<pad>']}])
    
    def post_process(self, corrected, original):
        # 对齐处理：优先保留原始文本中的正确部分
        min_len = min(len(corrected), len(original))
        final = []
        for oc, cc in zip(original[:min_len], corrected[:min_len]):
            final.append(oc if oc == cc else cc)
        return ''.join(final) + original[min_len:]  # 保留超长原始部分


