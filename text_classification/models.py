from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

class TextClassifier:
    def __init__(self, model_type='nb'):
        if model_type == 'nb':
            self.model = MultinomialNB()
        elif model_type == 'knn':
            self.model = KNeighborsClassifier(n_neighbors=5)
        else:
            raise ValueError("model_type must be 'nb' or 'knn'")
            
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred)
        return report
        
    def predict(self, X):
        return self.model.predict(X) 