import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM
from typing import List, Dict, Any
from tqdm import tqdm

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


# import torch
# import torch.nn as nn
# from typing import List, Dict, Any
# from tqdm import tqdm
# # 字符级别的 LSTM 模型
# class CharLevelLSTM(nn.Module):
#     def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256, device='cpu'):
#         super(CharLevelLSTM, self).__init__()
#         self.device = device
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, vocab_size)
    
#     def forward(self, x):
#         x = self.embedding(x)
#         lstm_out, _ = self.lstm(x)
#         out = self.fc(lstm_out)
#         return out

# # 纠错模型
# class NNCorrector:
#     def __init__(self, train_data: List[Dict[str, Any]], device='cpu', embedding_dim=128, hidden_dim=256):
#         self.device = device
#         self.vocab, self.reverse_vocab = self.build_vocab(train_data)
#         self.model = CharLevelLSTM(vocab_size=len(self.vocab), embedding_dim=embedding_dim, hidden_dim=hidden_dim, device=device).to(self.device)
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
#         self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.vocab["<pad>"])  # Ignore padding index
    
#     def build_vocab(self, train_data: List[Dict[str, Any]]) -> Dict[str, int]:
#         # 创建词汇表（字符及其索引）
#         vocab = {"<pad>": 0, "<unk>": 1}  # <pad> 是填充符，<unk> 是未知字符
#         for batch in train_data:
#             for text in [batch['source'], batch['target']]:
#                 for char in text:
#                     if char not in vocab:
#                         vocab[char] = len(vocab)
#         reverse_vocab = {v: k for k, v in vocab.items()}  # 创建反向映射
#         return vocab, reverse_vocab
    
#     def train_model(self, train_data: List[Dict[str, Any]]) -> None:
#         self.model.train()
#         for batch in tqdm(train_data):
#             # Tokenize the source and target text (convert chars to indices)
#             inputs = [self.vocab.get(c, self.vocab["<unk>"]) for c in batch['source']]
#             targets = [self.vocab.get(c, self.vocab["<unk>"]) for c in batch['target']]

#             # Convert inputs and targets to tensors
#             inputs_tensor = torch.tensor(inputs).unsqueeze(0).to(self.device)  # Add batch dimension
#             targets_tensor = torch.tensor(targets).unsqueeze(0).to(self.device)
            
#             # Ensure inputs and targets have the same length (pad if necessary)
#             max_length = max(inputs_tensor.size(1), targets_tensor.size(1))
#             inputs_tensor = nn.functional.pad(inputs_tensor, (0, max_length - inputs_tensor.size(1)))
#             targets_tensor = nn.functional.pad(targets_tensor, (0, max_length - targets_tensor.size(1)))

#             self.optimizer.zero_grad()

#             # Forward pass
#             outputs = self.model(inputs_tensor)
            
#             # Compute loss
#             loss = self.loss_fn(outputs.view(-1, len(self.vocab)), targets_tensor.view(-1))
#             loss.backward()
#             self.optimizer.step()

#     def correct(self, text: str) -> str:
#         self.model.eval()
        
#         # Tokenize the input text (convert chars to indices)
#         inputs = [self.vocab.get(c, self.vocab["<unk>"]) for c in text]
#         inputs_tensor = torch.tensor(inputs).unsqueeze(0).to(self.device)  # Add batch dimension
        
#         with torch.no_grad():
#             outputs = self.model(inputs_tensor)
        
#         # Convert output indices back to characters
#         predicted_indices = torch.argmax(outputs, dim=-1).squeeze(0).cpu().numpy()
#         corrected_text = ''.join([self.reverse_vocab[i] for i in predicted_indices])
        
#         return corrected_text


# import torch
# import torch.nn as nn
# from typing import List, Dict, Any
# from tqdm import tqdm

# # Transformer 模型
# class TransformerModel(nn.Module):
#     def __init__(self, vocab_size: int, embedding_dim: int = 128, num_heads: int = 8, num_layers: int = 6, hidden_dim: int = 512, device='cpu'):
#         super(TransformerModel, self).__init__()
#         self.device = device
        
#         # 词嵌入层
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
#         # Transformer Encoder-Decoder
#         self.transformer = nn.Transformer(
#             d_model=embedding_dim,
#             nhead=num_heads,
#             num_encoder_layers=num_layers,
#             num_decoder_layers=num_layers,
#             dim_feedforward=hidden_dim,
#             batch_first=True
#         )
        
#         # 输出层，预测词汇
#         self.fc_out = nn.Linear(embedding_dim, vocab_size)
    
#     def forward(self, src, tgt):
#         src_emb = self.embedding(src)  # 输入的源文本嵌入
#         tgt_emb = self.embedding(tgt)  # 目标文本嵌入
        
#         # Transformer 需要源文本和目标文本
#         output = self.transformer(src_emb, tgt_emb)
        
#         # 最后一层的输出
#         output = self.fc_out(output)
#         return output


# class NNCorrector:
#     def __init__(self, train_data: List[Dict[str, Any]], device='cpu', embedding_dim=128, num_heads=8, num_layers=6, hidden_dim=512):
#         self.device = device
#         self.vocab, self.reverse_vocab = self.build_vocab(train_data)
#         self.model = TransformerModel(vocab_size=len(self.vocab), embedding_dim=embedding_dim, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim, device=device).to(self.device)
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
#         self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.vocab["<pad>"])  # Ignore padding index
    
#     def build_vocab(self, train_data: List[Dict[str, Any]]) -> Dict[str, int]:
#         # 创建词汇表（字符及其索引）
#         vocab = {"<pad>": 0, "<unk>": 1}  # <pad> 是填充符，<unk> 是未知字符
#         for batch in train_data:
#             for text in [batch['source'], batch['target']]:
#                 for char in text:
#                     if char not in vocab:
#                         vocab[char] = len(vocab)
#         reverse_vocab = {v: k for k, v in vocab.items()}  # 创建反向映射
#         return vocab, reverse_vocab
    
#     def train_model(self, train_data: List[Dict[str, Any]]) -> None:
#         self.model.train()
#         for batch in tqdm(train_data):
#             # Tokenize the source and target text (convert chars to indices)
#             inputs = [self.vocab.get(c, self.vocab["<unk>"]) for c in batch['source']]
#             targets = [self.vocab.get(c, self.vocab["<unk>"]) for c in batch['target']]

#             # Convert inputs and targets to tensors
#             inputs_tensor = torch.tensor(inputs).unsqueeze(0).to(self.device)  # Add batch dimension
#             targets_tensor = torch.tensor(targets).unsqueeze(0).to(self.device)
            
#             # Ensure inputs and targets have the same length (pad if necessary)
#             max_length = max(inputs_tensor.size(1), targets_tensor.size(1))
#             inputs_tensor = nn.functional.pad(inputs_tensor, (0, max_length - inputs_tensor.size(1)))
#             targets_tensor = nn.functional.pad(targets_tensor, (0, max_length - targets_tensor.size(1)))

#             self.optimizer.zero_grad()

#             # Forward pass
#             outputs = self.model(inputs_tensor, targets_tensor[:, :-1])  # Exclude last token from target as decoder input
            
#             # Compute loss
#             loss = self.loss_fn(outputs.view(-1, len(self.vocab)), targets_tensor[:, 1:].view(-1))  # Skip the first token in target
#             loss.backward()
#             self.optimizer.step()

#     def correct(self, text: str) -> str:
#         self.model.eval()
        
#         # Tokenize the input text (convert chars to indices)
#         inputs = [self.vocab.get(c, self.vocab["<unk>"]) for c in text]
#         inputs_tensor = torch.tensor(inputs).unsqueeze(0).to(self.device)  # Add batch dimension
        
#         # For correction, we need to generate the target sequence (start with a token)
#         generated_tokens = []
#         tgt_tensor = torch.tensor([self.vocab.get("<pad>")]).unsqueeze(0).to(self.device)  # Start token for decoder
        
#         with torch.no_grad():
#             # Run the model iteratively to generate output
#             for _ in range(len(inputs_tensor[0])):
#                 outputs = self.model(inputs_tensor, tgt_tensor)  # Forward pass
#                 next_token = torch.argmax(outputs[:, -1, :], dim=-1)  # Predict next token
#                 generated_tokens.append(next_token.item())
#                 tgt_tensor = torch.cat((tgt_tensor, next_token.unsqueeze(0)), dim=1)  # Append predicted token to target
        
#         # Convert output indices back to characters
#         corrected_text = ''.join([self.reverse_vocab[i] for i in generated_tokens])
        
#         return corrected_text



# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from collections import defaultdict
# import random
# from tqdm import tqdm

# class ImprovedCorrector(nn.Module):
#     def __init__(self, char2idx, device='cuda' if torch.cuda.is_available() else 'cpu'):
#         super().__init__()
#         self.device = device
#         self.char2idx = char2idx
#         self.idx2char = {v: k for k, v in char2idx.items()}
#         self.vocab_size = len(char2idx)
        
#         # 增强的模型架构
#         self.embedding = nn.Embedding(self.vocab_size, 256)
#         self.encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=256,
#                 nhead=8,
#                 dim_feedforward=1024,
#                 dropout=0.1
#             ),
#             num_layers=4
#         )
#         self.decoder = nn.LSTM(256, 512, bidirectional=True, num_layers=2, dropout=0.2)
#         self.fc = nn.Sequential(
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.LayerNorm(512),
#             nn.Linear(512, self.vocab_size)
#         )
        
#     def forward(self, x):
#         x = self.embedding(x)
#         x = self.encoder(x.permute(1, 0, 2))  # Transformer需要序列维度在前
#         x, _ = self.decoder(x.permute(1, 0, 2))
#         return self.fc(x)

# class CorrectorDataset(Dataset):
#     def __init__(self, data, char2idx, noise_prob=0.3):
#         self.data = data
#         self.char2idx = char2idx
#         self.noise_prob = noise_prob
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         clean = self.data[idx]['target']
#         # 数据增强：动态生成带噪声的输入
#         noisy = self.add_noise(clean)
#         return self.text_to_ids(noisy), self.text_to_ids(clean)
    
#     def add_noise(self, text):
#         # 多种噪声类型：增删改
#         chars = list(text)
#         if random.random() < self.noise_prob and len(chars) > 1:
#             noise_type = random.choice(['delete', 'replace', 'insert'])
#             if noise_type == 'delete':
#                 del_idx = random.randint(0, len(chars)-1)
#                 del chars[del_idx]
#             elif noise_type == 'replace':
#                 rep_idx = random.randint(0, len(chars)-1)
#                 chars[rep_idx] = random.choice(list(self.char2idx.keys()))
#             elif noise_type == 'insert':
#                 ins_idx = random.randint(0, len(chars))
#                 chars.insert(ins_idx, random.choice(list(self.char2idx.keys())))
#         return ''.join(chars)
    
#     def text_to_ids(self, text):
#         return [self.char2idx.get(c, self.char2idx['<unk>']) for c in text]

# class NNCorrector:
#     def __init__(self, train_data, device='cuda' if torch.cuda.is_available() else 'cpu'):
#         self.device = device
#         self.char2idx = self.build_vocab(train_data)
#         self.idx2char = {v: k for k, v in self.char2idx.items()}  # Add this line to create the reverse mapping
#         self.model = ImprovedCorrector(self.char2idx, device).to(device)
#         self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
#         self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=3)
#         self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.char2idx['<pad>'])
#         self.vocab_size = len(self.char2idx)

#     def build_vocab(self, data):
#         char_counts = defaultdict(int)
#         for item in data:
#             for c in item['target']:
#                 char_counts[c] += 1
#         chars = ['<pad>', '<unk>', '<sos>', '<eos>'] + [c for c in char_counts if char_counts[c] > 1]
#         return {c: i for i, c in enumerate(chars)}
    
#     def collate_fn(self, batch):
#         inputs, targets = zip(*batch)
#         max_len = max(max(len(x) for x in inputs), max(len(x) for x in targets))
        
#         def pad_seq(seqs):
#             padded = torch.full((len(seqs), max_len), self.char2idx['<pad>'], dtype=torch.long)
#             for i, seq in enumerate(seqs):
#                 padded[i, :len(seq)] = torch.tensor(seq)
#             return padded
        
#         return pad_seq(inputs), pad_seq(targets)
    
#     def train_model(self, train_data, epochs=2, batch_size=32):
#         dataset = CorrectorDataset(train_data, self.char2idx)
#         loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        
#         best_loss = float('inf')
#         for epoch in range(epochs):
#             self.model.train()
#             total_loss = 0
#             for inputs, targets in tqdm(loader, desc=f'Epoch {epoch+1}'):
#                 inputs = inputs.to(self.device)
#                 targets = targets.to(self.device)
                
#                 self.optimizer.zero_grad()
#                 outputs = self.model(inputs)
                
#                 loss = self.loss_fn(outputs.view(-1, self.vocab_size), targets.view(-1))
#                 loss.backward()
#                 nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
#                 self.optimizer.step()
                
#                 total_loss += loss.item()
            
#             avg_loss = total_loss / len(loader)
#             self.scheduler.step(avg_loss)
            
#             if avg_loss < best_loss:
#                 best_loss = avg_loss
#                 torch.save(self.model.state_dict(), 'best_model.pth')
    
#     def correct(self, text: str, noise_prob: float = 0.2) -> str:
#         self.model.eval()
#         with torch.no_grad():
#             # 创建临时数据集实例用于添加噪声
#             temp_dataset = CorrectorDataset([], self.char2idx, noise_prob=noise_prob)
#             noisy = temp_dataset.add_noise(text)  # 正确的噪声生成方式
            
#             input_ids = self.text_to_ids(noisy)
#             input_tensor = torch.tensor([input_ids], device=self.device)
            
#             outputs = self.model(input_tensor)
#             pred_ids = outputs.argmax(-1).squeeze().cpu().tolist()
            
#             corrected = self.ids_to_text(pred_ids)
#             legal_chars = set(self.char2idx.keys()) - {'<unk>', '<pad>'}
#             corrected = ''.join(c for c in corrected if c in legal_chars)
#             return self.post_process(corrected, original=text)

#     def text_to_ids(self, text):
#         return [self.char2idx.get(c, self.char2idx['<unk>']) for c in text]
    
#     def ids_to_text(self, ids):
#         return ''.join([self.idx2char.get(i, '<unk>') for i in ids if i not in {self.char2idx['<pad>']}])
    
#     def post_process(self, corrected, original):
#         # 对齐处理：优先保留原始文本中的正确部分
#         min_len = min(len(corrected), len(original))
#         final = []
#         for oc, cc in zip(original[:min_len], corrected[:min_len]):
#             final.append(oc if oc == cc else cc)
#         return ''.join(final) + original[min_len:]  # 保留超长原始部分





# class NNCorrector:
#     def __init__(self):
#         pass

#     def train_model(self, train_data: List[Dict[str, Any]]) -> None:
#         pass

#     def correct(self, text: str) -> str:
#         pass

# 由于任务较难，建议使用深度学习方法获得更好得分。

# 注意：

# 禁止使用LLM纠错
# 禁止使用现成的文本纠错工具和已经训练完成文本纠错模型
# 数据输入格式为：{"source": "敬请关注。", "target": "敬请关注。", "label": 0}
# 不知道违规不，但好用。