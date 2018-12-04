from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix


class NeuralNetworkClassifier(object):
    def __init__(self, alpha, hidden_layer_sizes, solver='adam', learning_rate='constant', activation='relu', tol=1e-4):
        self.mlp = MLPClassifier(solver=solver,
                                 alpha=alpha,
                                 hidden_layer_sizes=hidden_layer_sizes,
                                 learning_rate=learning_rate,
                                 activation=activation,
                                 tol=tol)

    def train(self, x_train, y_train):
        print('Classifier training in process...')
        self.mlp.fit(x_train, y_train)
        print('Classifier training concluded')

    def test(self, x_test, y_test):
        print('Testing classifier')
        predictions = self.mlp.predict(x_test)
        confusion_matrix(y_test, predictions)  # no se si se ocupa
        return classification_report(y_test, predictions)
