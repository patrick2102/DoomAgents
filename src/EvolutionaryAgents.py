import numpy as np
import matplotlib.pyplot as plt
from vizdoom.vizdoom import DoomGame
from deap import base
from deap import creator
from deap import tools

from src.Agent import AgentBase


class EvolutionaryAgentSimple(AgentBase):
    def __init__(self, x_size, y_size, actions, dims=3):
        super().__init__()
        self.actions = actions
        self.gauss_mutation_chance = 0.5
        self.x_s = x_size
        self.y_s = y_size
        self.dims = dims

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        self.toolbox.register("individual", self.network, creator.Individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.eval)
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def network(self, individual):
        state_space = self.x_s * self.y_s * self.dims
        l1_out = 32
        l1 = np.random.rand(state_space, l1_out)
        l2 = np.random.rand(l1_out, len(self.actions))
        layers = [l1, l2]
        return individual(layers)

    def eval(self, individual, game: DoomGame):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            action = self.get_action(individual, state)
            game.make_action(action)
        return game.get_total_reward(),

    def get_action(self, individual, state):
        state = state.screen_buffer
        state = state.flatten()
        individual = np.array(individual)
        mut = np.matmul(individual[0], individual[1])
        return self.actions[np.argmax(np.matmul(mut.T, state))]


    def set_available_actions(self, actions):
        self.actions = actions

    def fitness_statistics(self, pop):
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        fits.sort()
        mid = fits[int(len(fits) / 2)]
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        # print("  Avg %s" % mean)
        print("  Median %s" % mid)
        print("  Std %s" % std)
        return fits, mean, mid, std

    def train(self, n, g_limit, doomGame):
        toolbox = self.toolbox
        pop = self.toolbox.population(n)

        plt.ion()
        plt.show()
        plt.xlabel('generation')
        plt.ylabel('median score')
        plt.title('Doom')
        x = []
        y = []

        for i in range(n):
            fit = toolbox.evaluate(pop[i], doomGame)
            pop[i].fitness.values = fit

        for g in range(g_limit):
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            for i in range(n):
                fit = toolbox.evaluate(offspring[i], doomGame)
                offspring[i].fitness.values = fit

            pop[:] = offspring

            fits, mean, median, std = self.fitness_statistics(pop)

            x.append(g)
            y.append(fits[int(len(fits)/2)])
            #y.append(sum(fits) / len(fits))
            plt.plot(x, y)

            plt.draw()
            plt.pause(0.1)

