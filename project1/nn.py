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
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM
from typing import List, Dict, Any
from tqdm import tqdm

class NNCorrector:
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
