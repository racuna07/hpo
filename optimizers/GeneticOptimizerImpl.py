from pandas import DataFrame
from threading import Thread
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
