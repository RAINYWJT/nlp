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
# import torch
# import torch.nn as nn
# from transformers import BertTokenizer, BertForMaskedLM
# from typing import List, Dict, Any
# from tqdm import tqdm

# class NNCorrector:
#     def __init__(self, model_name='bert-base-chinese', device='cpu'):
#         self.device = device
#         self.tokenizer = BertTokenizer.from_pretrained(model_name)
#         self.encoder = BertForMaskedLM.from_pretrained(model_name).to(self.device)
#         self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-5)
#         self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

#     def train_model(self, train_data: List[Dict[str, Any]]) -> None:
#         self.encoder.train()
#         for batch in tqdm(train_data):
#             # Tokenize the source and target text
#             inputs = self.tokenizer(batch['source'], padding=True, truncation=True, return_tensors='pt').input_ids.to(self.device)
#             targets = self.tokenizer(batch['target'], padding=True, truncation=True, return_tensors='pt').input_ids.to(self.device)

#             if inputs.size(1) != targets.size(1):  # Compare sequence length, not batch size
#                 # Make sure inputs and targets have the same length (pad shorter one)
#                 max_length = max(inputs.size(1), targets.size(1))
#                 inputs = nn.functional.pad(inputs, (0, max_length - inputs.size(1)))
#                 targets = nn.functional.pad(targets, (0, max_length - targets.size(1)))

#             # print(inputs.shape , targets.shape)
#             assert inputs.size(0) == targets.size(0)
            
#             self.optimizer.zero_grad()
            
#             # Forward pass
#             outputs = self.encoder(input_ids=inputs, labels=targets)
#             loss = outputs.loss
#             loss.backward()
#             self.optimizer.step()

#     def correct(self, text: str) -> str:
#         self.encoder.eval()
#         # Tokenize the input text
#         inputs = self.tokenizer(text, return_tensors='pt').input_ids.to(self.device)
        
#         with torch.no_grad():
#             outputs = self.encoder(input_ids=inputs)
        
#         logits = outputs.logits
#         predicted_ids = torch.argmax(logits, dim=-1)
#         corrected_text = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
#         corrected_text = corrected_text.replace(" ", "")
#         return corrected_text


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
