import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from typing import List, Dict, Any
from collections import Counter

class NNCorrector(nn.Module):
    def __init__(self):
        super(NNCorrector, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size,
                            hidden_size=256,
                            num_layers=2,
                            bidirectional=True,
                            batch_first=True)
        self.fc = nn.Linear(512, 2)  # 二分类问题，0 为正确，1 为错误
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state
        lstm_out, _ = self.lstm(sequence_output)
        out = self.fc(lstm_out[:, -1, :])
        return out

    def train_model(self, train_data: List[Dict[str, Any]], epochs=5, batch_size=32):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        loss_fn = nn.CrossEntropyLoss()

        # 设置模型为训练模式
        self.train()

        for epoch in range(epochs):
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            for batch_idx in range(0, len(train_data), batch_size):
                batch = train_data[batch_idx:batch_idx+batch_size]
                source_texts = [item["source"] for item in batch]
                target_texts = [item["target"] for item in batch]
                labels = [item["label"] for item in batch]

                encodings = self.tokenizer(source_texts, padding=True, truncation=True, return_tensors="pt")
                input_ids = encodings["input_ids"].to(self.device)
                attention_mask = encodings["attention_mask"].to(self.device)

                outputs = self(input_ids, attention_mask)
                loss = loss_fn(outputs, torch.tensor(labels).to(self.device))
                total_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                correct_predictions += (predictions == torch.tensor(labels).to(self.device)).sum().item()
                total_predictions += len(labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_data)}, Accuracy: {correct_predictions/total_predictions}")

    def correct(self, text: str) -> str:
        # 设置模型为评估模式
        self.eval()
        encodings = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        
        with torch.no_grad():
            output = self(input_ids, attention_mask)
            prediction = torch.argmax(output, dim=1).item()
        
        if prediction == 1:
            return "纠正后的文本"  # 在此处添加具体的纠错逻辑
        else:
            return text