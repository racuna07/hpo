from data import data_utils
from classifiers import neural_network_classifier
from optimizers import GeneticOptimizerImpl as genetic


def main():
    results = genetic.get_fitness_from_turtle([0,1,0,0,1,0,0,1,1,1,0,0,0,1,0])
    print("Fertig!")

if __name__ == '__main__':
    main()
