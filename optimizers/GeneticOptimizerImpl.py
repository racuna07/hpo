from pandas import DataFrame
from threading import Thread
from classifiers import neural_network_classifier as nnc
from data import data_utils as data_utils
import pyNetLogo
import random

# CONSTANTS
FITNESS_THRESHOLD = 115
MAX_ITER = 100
POPULATION_SIZE = 100
MUTATION_RATE = 70
CROSSOVER_RATE = 0.01

netlogo = pyNetLogo.NetLogoLink(gui=True, netlogo_version='6')


def setup():
    netlogo.command('clear-all')
    netlogo.command('set population-size ' + str(POPULATION_SIZE))
    netlogo.command('set mutation-rate ' + str(MUTATION_RATE))
    netlogo.command('set crossover-rate ' + str(CROSSOVER_RATE))
    netlogo.command('setup')


def main():
    netlogo.load_model('C:\\Users\\Rodrigo\\PycharmProjects\\hpo\\resources\\Simple Genetic Algorithm.nlogo')
    setup()
    # Evaluate initial fitness
    who = netlogo.report('map [s -> [who] of s] sort turtles')
    turtles_bits_dict = {int(i): netlogo.report("[bits] of turtle " + str(i)) for i in who}
    fitness_frame = get_fitness_frame(who, turtles_bits_dict)
    max_fitness = fitness_frame['fitness'].max()
    # Update turtles initial fitness in netlogo
    netlogo.write_NetLogo_attriblist(fitness_frame[['who', 'fitness']], 'turtle')
    iteration_number = 1
    # iterate until condition is met
    while not should_stop(iteration_number, max_fitness):
        max_fitness = go()
        print("Iteration: {},\t max_fitness: {}".format(iteration_number,max_fitness))
        iteration_number += 1
    netlogo.kill_workspace()


def should_stop(iteration, max_fitness):
    return iteration == MAX_ITER or max_fitness > FITNESS_THRESHOLD


def save_fitness(bits, i, fitness_array):
    fitness_array[i] = 88 - random.randint(0, 10)


def get_fitness_frame(who, turtles_bits_dict):
    dictionary = dict()
    fitness = [None] * POPULATION_SIZE
    threads = [Thread(target=save_fitness, args=(turtles_bits_dict[who[i]], i, fitness)) for i in range(len(who))]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    dictionary['who'] = who
    dictionary['fitness'] = fitness
    return DataFrame(dictionary)


def go():
    # Create new generation
    netlogo.command('create-next-generation')
    # Retrieve individuals from netlogo
    who = netlogo.report('map [s -> [who] of s] sort turtles')
    turtles_bits_dict = {int(i): netlogo.report("[bits] of turtle " + str(i)) for i in who}
    # Evaluate new generation
    fitness_frame = get_fitness_frame(who, turtles_bits_dict)
    max_fitness = fitness_frame['fitness'].max()
    # Update turtles fitness in netlogo
    netlogo.write_NetLogo_attriblist(fitness_frame[['who', 'fitness']], 'turtle')
    # Update view
    netlogo.command('update-display')
    netlogo.command('tick')
    return max_fitness


if __name__ == '__main__':
    main()


def get_fitness_from_turtle(array):
    a = array[0]
    b = array[1]
    # First get the activation function
    act_fnc_num = get_number_from_array([a,b])
    if act_fnc_num == 0:
        act_fnc = "identity"
    elif act_fnc_num == 1:
        act_fnc = "logistic"
    elif act_fnc_num == 2:
        act_fnc = "tanh"
    else: act_fnc = "relu"
    # Get the learning rate
    a = array[2]
    b = array[3]
    lrn_rate_num = get_number_from_array([a,b])
    if lrn_rate_num == 0:
        lrn_rate = "constant"
    elif lrn_rate_num == 1:
        lrn_rate = "invscaling"
    elif lrn_rate_num == 2:
        lrn_rate = "adaptative"
    else: return 0
    # Get the solver
    a = array[4]
    b = array[5]
    solver_num = get_number_from_array([a,b])
    if solver_num == 0:
        solver = "lbfgs"
    elif solver_num == 1:
        solver = "sgd"
    elif solver_num == 2:
        solver = "adam"
    else: return 0
    # And now the alpha parameter
    a = array[6]
    b = array[7]
    c = array[8]
    alpha_num = get_number_from_array([a,b,c])
    if alpha_num == 0:
        alpha = 1e-5
    elif alpha_num == 1:
        alpha = 1e-3
    elif alpha_num == 2:
        alpha = 1e-1
    elif alpha_num == 3:
        alpha = 1e1
    elif alpha_num == 4:
        alpha = 1e3
    else: return 0
    #And now the number of layers
    a = array[9]
    b = array[10]
    c = array[11]
    layer_num = get_number_from_array([a, b, c])
    if layer_num <= 4:
        layers = layer_num + 1
    else:
        return 0
    # Finally the number of intermediate layers
    a = array[12]
    b = array[13]
    c = array[14]
    neurons_layer_num = get_number_from_array([a, b, c])
    if neurons_layer_num <= 4:
        neurons_layer = (neurons_layer_num + 1)*10
    else:
        return 0
    #To this point we have all the parameters
    hidden_layer_sizes = []
    i = 0
    for i in range(0,layers):
        hidden_layer_sizes.append(neurons_layer)

    clasiffier = nnc.NeuralNetworkClassifier(alpha, hidden_layer_sizes, solver, lrn_rate, act_fnc)
    du = data_utils.DataUtils()
    x_train, y_train = du.get_training_data()
    x_test, y_test = du.get_testing_data()
    clasiffier.train(x_train, y_train)
    results = clasiffier.test(x_test, y_test)
    return results


def get_number_from_array(array):
    length = len(array)
    tend = length - 1
    number = 0
    for i in range(0, length):
        number += array[tend - i]*(2**i)
    return number
