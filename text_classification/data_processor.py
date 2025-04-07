import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self):
        self.categories = ["体育", "家居", "房产", "教育", "科技", "财经"]
        self.vectorizer = TfidfVectorizer(max_features=5000)
        
    def load_data(self, file_path):
        texts = []
        labels = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                label, text = line.strip().split('\t')
                texts.append(' '.join(jieba.cut(text)))
                labels.append(self.categories.index(label))
        return texts, np.array(labels)
    
    def process_data(self, file_path):
        texts, labels = self.load_data(file_path)
        
        # 划分训练集、验证集和测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, test_size=0.1, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.111, random_state=42)
        
        # TF-IDF特征提取
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_val_tfidf = self.vectorizer.transform(X_val)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        return (X_train_tfidf, y_train), (X_val_tfidf, y_val), (X_test_tfidf, y_test) 