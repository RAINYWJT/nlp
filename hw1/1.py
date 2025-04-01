import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

nltk.download('punkt')

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim, negative_samples=3):
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.negative_samples = negative_samples
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, target_idx, context_idx, negative_idx):
        batch_size = target_idx.shape[0]
        target_embed = self.embeddings(target_idx)
        context_embed = self.embeddings(context_idx)
        neg_embed = self.embeddings(negative_idx)

        pos_loss = F.logsigmoid(torch.sum(target_embed * context_embed, dim=1))
        neg_loss = F.logsigmoid(-torch.sum(target_embed.unsqueeze(1) * neg_embed, dim=2)).sum(dim=1)

        return -torch.mean(pos_loss + neg_loss)

def generate_skipgram_data(window_size, tokenized_sentences, word2index, vocab_size, num_neg_samples=3):
    training_data = []
    
    for sentence in tokenized_sentences:
        indexed_sentence = [word2index[word] for word in sentence]
        
        for i, target_idx in enumerate(indexed_sentence):
            # 获取上下文词索引（正样本）
            context_indices = [
                indexed_sentence[j]
                for j in range(i - window_size, i + window_size + 1)
                if 0 <= j < len(indexed_sentence) and j != i
            ]

            # 负采样：确保负样本不在上下文中，也不等于目标词
            negative_samples = set()
            while len(negative_samples) < num_neg_samples:
                neg_idx = random.randint(0, vocab_size - 1)
                if neg_idx != target_idx and neg_idx not in context_indices:
                    negative_samples.add(neg_idx)

            training_data.append((target_idx, context_indices, list(negative_samples)))

    return training_data

def main():
    text_data = """
    Natural language processing (NLP) is a field of artificial intelligence that deals with the interaction 
    between computers and humans using natural language. The ultimate goal of NLP is to enable computers 
    to understand, interpret, and generate human language in a way that is valuable.
    """
    
    sentences = nltk.sent_tokenize(text_data)
    tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
    vocab = set(word for sentence in tokenized_sentences for word in sentence)
    vocab_size = len(vocab)
    word2index = {word: i for i, word in enumerate(vocab)}
    
    window_size = 2
    training_data = generate_skipgram_data(window_size, tokenized_sentences, word2index, vocab_size)
    print(training_data)
    # assert 0
    
    embedding_dim = 10
    learning_rate = 0.01
    epochs = 200
    
    model = SkipGram(vocab_size, embedding_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        total_loss = 0
        for target_idx, context_idx, negative_samples in training_data:
            target_idx = torch.tensor([target_idx])
            context_idx = torch.tensor([context_idx])
            negative_samples = torch.tensor(negative_samples)

            optimizer.zero_grad()
            loss = model(target_idx, context_idx, negative_samples)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
    
    word_embeddings = np.array(model.embeddings.weight.tolist())
    tsne = TSNE(n_components=2, random_state=42)
    word_embeddings_2d = tsne.fit_transform(word_embeddings)
    
    plt.figure(figsize=(10, 10))
    for i, word in enumerate(vocab):
        plt.scatter(word_embeddings_2d[i, 0], word_embeddings_2d[i, 1])
        plt.annotate(word, (word_embeddings_2d[i, 0], word_embeddings_2d[i, 1]), fontsize=12)
    
    plt.title('Word Embeddings Visualization')
    plt.show()

if __name__ == "__main__":
    main()