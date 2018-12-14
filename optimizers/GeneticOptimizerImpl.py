from pandas import DataFrame
from classifiers import neural_network_classifier as nnc
from data import data_utils as data_utils
import pyNetLogo

# CONSTANTS
FITNESS_THRESHOLD = 0.95
MAX_ITER = 300
POPULATION_SIZE = 30
MUTATION_RATE = 70
CROSSOVER_RATE = 0.02

netlogo = pyNetLogo.NetLogoLink(gui=True, netlogo_version='6')
du = data_utils.DataUtils()
cache = dict()


def setup():
    netlogo.command('clear-all')
    netlogo.command('set population-size ' + str(POPULATION_SIZE))
    netlogo.command('set mutation-rate ' + str(MUTATION_RATE))
    netlogo.command('set crossover-rate ' + str(CROSSOVER_RATE))
    netlogo.command('setup')


def main():
    global_max = 0
    netlogo.load_model('C:\\Users\\carlos\\PycharmProjects\\hpo\\resources\\Simple Genetic Algorithm.nlogo')
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
        print("Iteration: {},\t max_fitness: {}".format(iteration_number, max_fitness))
        if max_fitness > global_max:
            global_max = max_fitness
        print("Global Max: {}".format(global_max))
        iteration_number += 1
    netlogo.kill_workspace()


def should_stop(iteration, max_fitness):
    return iteration == MAX_ITER or max_fitness > FITNESS_THRESHOLD


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


def get_fitness_from_turtle(bits_array, i, fitness_array, du, cache):
    valid = True
    a = bits_array[0]
    b = bits_array[1]
    # First get the activation function
    act_fnc_num = get_number_from_array([a, b])
    if act_fnc_num == 0:
        act_fnc = "identity"
    elif act_fnc_num == 1:
        act_fnc = "logistic"
    elif act_fnc_num == 2:
        act_fnc = "tanh"
    else:
        act_fnc = "relu"
    # Get the learning rate
    a = bits_array[2]
    b = bits_array[3]
    lrn_rate_num = get_number_from_array([a, b])
    if lrn_rate_num == 0:
        lrn_rate = "constant"
    elif lrn_rate_num == 1:
        lrn_rate = "invscaling"
    elif lrn_rate_num == 2:
        lrn_rate = "adaptative"
    else:
        valid = False
    # Get the solver
    a = bits_array[4]
    b = bits_array[5]
    solver_num = get_number_from_array([a, b])
    if solver_num == 0:
        solver = "lbfgs"
    elif solver_num == 1:
        solver = "sgd"
    elif solver_num == 2:
        solver = "adam"
    else:
        valid = False
    # And now the alpha parameter
    a = bits_array[6]
    b = bits_array[7]
    c = bits_array[8]
    alpha_num = get_number_from_array([a, b, c])
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
    else:
        valid = False
    # And now the number of layers
    a = bits_array[9]
    b = bits_array[10]
    c = bits_array[11]
    layer_num = get_number_from_array([a, b, c])
    if layer_num <= 5:
        layers = layer_num + 1
    else:
        layers = layer_num + 1
        valid = False
    # Finally the number of intermediate layers
    a = bits_array[12]
    b = bits_array[13]
    c = bits_array[14]
    neurons_layer_num = get_number_from_array([a, b, c])
    if neurons_layer_num <= 5:
        neurons_layer = (neurons_layer_num + 1) * 10
    else:
        neurons_layer = (neurons_layer_num + 1) * 10
        valid = False
    # To this point we have all the parameters
    hidden_layer_sizes = []
    for i in range(0, layers):
        hidden_layer_sizes.append(neurons_layer)

    # Check cache
    individual = str([act_fnc_num, lrn_rate_num, solver_num, alpha_num, layer_num, neurons_layer_num])
    if cache.__contains__(individual):
        fitness_array[i] = cache[individual]
        return
    else:
        result = 0
        if valid:
            try:
                classifier = nnc.NeuralNetworkClassifier(alpha, hidden_layer_sizes, solver, lrn_rate, act_fnc)
                x_train, y_train = du.get_training_data()
                x_test, y_test = du.get_testing_data()
                classifier.train(x_train, y_train)
                result = classifier.test(x_test, y_test)
                fitness_array[i] = result
                if result > 0.7:
                    print(individual, result)
            except ValueError:
                pass
        cache[individual] = result


def get_number_from_array(array):
    length = len(array)
    tend = length - 1
    number = 0
    for i in range(0, length):
        number += array[tend - i] * (2 ** i)
    return int(number)


def get_fitness_frame(who, turtles_bits_dict):
    dictionary = dict()
    fitness = [0] * POPULATION_SIZE
    for i in range(len(who)):
        get_fitness_from_turtle(turtles_bits_dict[who[i]], i, fitness, du, cache)
    dictionary['who'] = who
    dictionary['fitness'] = fitness
    return DataFrame(dictionary)


if __name__ == '__main__':
    main()
