from data import data_utils
from classifiers import neural_network_classifier


def main():
    du = data_utils.DataUtils()
    classifier = neural_network_classifier.NeuralNetworkClassifier(solver='adam',
                                                                   alpha=1e-3,
                                                                   hidden_layer_sizes=(30, 40))
    x_train, y_train = du.get_training_data()
    x_test, y_test = du.get_testing_data()

    classifier.train(x_train, y_train)
    results = classifier.test(x_test, y_test)
    print(results)


if __name__ == '__main__':
    main()
