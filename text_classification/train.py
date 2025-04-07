from data_processor import DataProcessor
from models import TextClassifier
import joblib

def main():
    # 数据处理
    processor = DataProcessor()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.process_data('filtered_cnews.train.txt')
    
    # 训练朴素贝叶斯模型
    print("训练朴素贝叶斯模型...")
    nb_classifier = TextClassifier('nb')
    nb_classifier.train(X_train, y_train)
    print("\n朴素贝叶斯模型验证集评估结果:")
    print(nb_classifier.evaluate(X_val, y_val))
    
    # 训练KNN模型
    print("\n训练KNN模型...")
    knn_classifier = TextClassifier('knn')
    knn_classifier.train(X_train, y_train)
    print("\nKNN模型验证集评估结果:")
    print(knn_classifier.evaluate(X_val, y_val))
    
    # 保存模型
    joblib.dump(nb_classifier.model, 'nb_model.pkl')
    joblib.dump(knn_classifier.model, 'knn_model.pkl')
    joblib.dump(processor.vectorizer, 'vectorizer.pkl')

if __name__ == "__main__":
    main() 