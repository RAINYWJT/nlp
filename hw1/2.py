import numpy as np
from collections import defaultdict
from nltk.tokenize import word_tokenize
import nltk

# 下载 punkt 分词工具
nltk.download('punkt')

# 示例文本数据
text_data = """Natural language processing (NLP) is a field of artificial intelligence that deals with the interaction between computers and humans using natural language. The ultimate goal of NLP is to enable computers to understand, interpret, and generate human language in a way that is valuable. Word2Vec is a technique in NLP used to compute vector representations of words. These representations capture semantic meaning of words and are widely used for various NLP tasks such as word similarity, text classification, and machine translation."""

# 对文本进行分词
sentences = nltk.sent_tokenize(text_data)
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

# 构建词汇表
vocab = set(word for sentence in tokenized_sentences for word in sentence)
vocab_size = len(vocab)

# 为每个词分配一个唯一的整数ID
word2index = {word: i for i, word in enumerate(vocab)}
index2word = {i: word for word, i in word2index.items()}

# 生成训练样本 (target, context)
window_size = 2
training_data = []

for sentence in tokenized_sentences:
    for i, target_word in enumerate(sentence):
        target_idx = word2index[target_word]
        context_indices = list(range(max(0, i - window_size), min(len(sentence), i + window_size + 1)))
        context_indices.remove(i)  # 移除目标词索引
        for context_idx in context_indices:
            context_word = sentence[context_idx]
            training_data.append((target_idx, word2index[context_word]))

# Skip-gram模型参数
embedding_dim = 10  # 词向量维度
learning_rate = 0.01  # 学习率
epochs = 100  # 训练轮次

# 初始化词向量矩阵
W = np.random.randn(vocab_size, embedding_dim)
W_context = np.random.randn(vocab_size, embedding_dim)

def train_skip_gram(training_data, W, W_context, learning_rate, epochs):
    for epoch in range(epochs):
        total_loss = 0
        for target_idx, context_idx in training_data:
            # 正向传播
            v_target = W[target_idx]
            v_context = W_context[context_idx]
            score = np.dot(v_target, v_context)
            exp_score = np.exp(score)
            pred = exp_score / (1 + exp_score)

            # 计算损失
            loss = -np.log(pred)
            total_loss += loss

            # 反向传播更新参数
            grad = (pred - 1)  # 对应负采样
            W[target_idx] -= learning_rate * grad * v_context
            W_context[context_idx] -= learning_rate * grad * v_target

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}")

train_skip_gram(training_data, W, W_context, learning_rate, epochs)

def predict_word(word, W):
    word_idx = word2index[word]
    return W[word_idx]

# 示例预测
print("\nWord vector for 'language':", predict_word('language', W))
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Extract the word vectors from the W matrix
word_vectors = W

# Apply t-SNE to reduce to 2 dimensions
tsne = TSNE(n_components=2, random_state=42)
word_vectors_2d = tsne.fit_transform(word_vectors)

# Plot the 2D word embeddings
plt.figure(figsize=(10, 10))
for i, word in enumerate(vocab):
    x, y = word_vectors_2d[i]
    plt.scatter(x, y)
    plt.text(x + 0.01, y + 0.01, index2word[i], fontsize=12)

plt.title("t-SNE visualization of word embeddings")
plt.show()
