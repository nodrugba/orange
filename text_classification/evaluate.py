from data_processor import DataProcessor
from models import TextClassifier
import joblib

def main():
    # 加载模型和向量器
    nb_model = joblib.load('nb_model.pkl')
    knn_model = joblib.load('knn_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    
    # 数据处理
    processor = DataProcessor()
    processor.vectorizer = vectorizer
    (_, _), (_, _), (X_test, y_test) = processor.process_data('filtered_cnews.train.txt')
    
    # 评估朴素贝叶斯模型
    print("朴素贝叶斯模型测试集评估结果:")
    nb_classifier = TextClassifier('nb')
    nb_classifier.model = nb_model
    print(nb_classifier.evaluate(X_test, y_test))
    
    # 评估KNN模型
    print("\nKNN模型测试集评估结果:")
    knn_classifier = TextClassifier('knn')
    knn_classifier.model = knn_model
    print(knn_classifier.evaluate(X_test, y_test))

if __name__ == "__main__":
    main() 