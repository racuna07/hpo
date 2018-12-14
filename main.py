from classifiers import neural_network_classifier as nnc
from data import data_utils as data_utils

# Prueba con el proyecto del semestre pasado, precision = 0.8847049755663746
def main():
    du = data_utils.DataUtils()
    alpha = 1e-5
    hidden_layer_sizes = (30,40)
    solver = 'adam'
    lrn_rate = 'constant'
    act_fnc = 'relu'
    classifier = nnc.NeuralNetworkClassifier(alpha, hidden_layer_sizes, solver, lrn_rate, act_fnc)
    x_train, y_train = du.get_training_data()
    x_test, y_test = du.get_testing_data()
    classifier.train(x_train, y_train)
    result = classifier.test(x_test, y_test)
    if result > 0.7:
        print(result)


if __name__ == '__main__':
    main()
