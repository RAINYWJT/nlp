import numpy as np
import random
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

np.seterr('ignore')
nltk.download('punkt')
np.random.seed(0)

# 引入负采样
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim, negative_samples=3):
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.negative_samples = negative_samples

        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)

    def forward(self, target_idx, context_idx):
        target_embed = self.embeddings(target_idx)
        context_embed = self.embeddings(context_idx)
        loss = self.negative_sampling_loss(target_embed, context_embed)
        return loss

    # 负采样
    def negative_sampling_loss(self, target_embed, context_embed):
        pos_loss = F.logsigmoid(torch.sum(target_embed * context_embed, dim=1))
        neg_loss = 0
        for _ in range(self.negative_samples):
            # 由于没有那种真正负采样的词表，只能稍微random一下了，如果有比如
            # process 为true label
            # pro，processing 为负样本
            # 这样就好了，但是没办法对于这么长的一个语句去生成负样本
            neg_target = random.randint(0, self.vocab_size - 1)
            # print(neg_target)
            neg_target_embed = self.embeddings(torch.tensor([neg_target]))
            neg_loss += F.logsigmoid(-torch.sum(target_embed * neg_target_embed, dim=1))
        total_loss = -torch.mean(pos_loss + neg_loss)
        return total_loss


def generate_skipgram_data(window_size, tokenized_sentences, word2index):
    training_data = []
    for sentence in tokenized_sentences:
        for i, target_word in enumerate(sentence):
            target_idx = word2index[target_word]
            for j in range(i - window_size, i + window_size + 1):
                if j >= 0 and j < len(sentence) and i != j:
                    context_idx = word2index[sentence[j]]
                    training_data.append((target_idx, context_idx))
    return training_data

def main():
    # 示例文本数据
    text_data = """
    Natural language processing (NLP) is a field of artificial intelligence that deals with the interaction 
    between computers and humans using natural language. The ultimate goal of NLP is to enable computers 
    to understand, interpret, and generate human language in a way that is valuable.
    Word2Vec is a technique in NLP used to compute vector representations of words. These representations
    capture semantic meaning of words and are widely used for various NLP tasks such as word similarity, 
    text classification, and machine translation.
    """

    # 对文本进行分词
    sentences = nltk.sent_tokenize(text_data)
    tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

    # 构建词汇表
    vocab = set(word for sentence in tokenized_sentences for word in sentence)
    vocab_size = len(vocab)

    # 为每个词分配一个唯一的整数ID
    word2index = {word: i for i, word in enumerate(vocab)}

    # 生成训练数据
    window_size = 2
    training_data = generate_skipgram_data(window_size, tokenized_sentences, word2index)
    
    # 初始化词向量
    embedding_dim = 10
    learning_rate = 0.01
    epochs = 100 # 之前我用的200

    model = SkipGram(vocab_size, embedding_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        for target_idx, context_idx in training_data:
            target_idx = torch.tensor([target_idx])
            context_idx = torch.tensor([context_idx])

            optimizer.zero_grad()  
            loss = model(target_idx, context_idx) 

            loss.backward()  
            optimizer.step()  

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    # 预测词向量
    def predict_word(word):
        word_idx = word2index[word]
        return model.embeddings(torch.tensor([word_idx]))

    word_vector = predict_word('language')
    print("\nWord vector for 'language':", word_vector)

    # print(model.embeddings.weight, type(model.embeddings.weight))
    word_embeddings = np.array(model.embeddings.weight.tolist())
    print(word_embeddings.shape)

    # TSNE
    tsne = TSNE(n_components=2, random_state=42)
    word_embeddings_2d = tsne.fit_transform(word_embeddings)
    plt.figure(figsize=(10, 10))
    for i, word in enumerate(vocab):
        plt.scatter(word_embeddings_2d[i, 0], word_embeddings_2d[i, 1])
        plt.annotate(word, (word_embeddings_2d[i, 0], word_embeddings_2d[i, 1]), fontsize=12)
    
    plt.title('Word Embeddings')
    # plt.savefig('Word Embeddings window_size 3（1）.png')
    plt.show()


if __name__ == "__main__":
    main()
