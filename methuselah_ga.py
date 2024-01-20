import logging
from typing import Any, Iterable

import numpy as np

from constants import (
    CROSSOVER_RATE,
    MUTATION_RATE,
    GENERATIONS,
    MAX_POPULATION_CLUSTERS,
)
from game_of_life import GameOfLife

logger = logging.getLogger(__name__)


class MethuselahGA:
    def __init__(self, starting_population_size: int, board_size: int):
        self.starting_population_size = starting_population_size
        self.grid_size = board_size
        self.population = self.initialize_population()

        self._crossover_rate = CROSSOVER_RATE
        self._mutation_rate = MUTATION_RATE

        self._avg_fitnesses = []

    def get_best_configuration(self):
        self.start()

        final_configurations_fitnesses = {
            self.evaluate(individual): individual for individual in self.population
        }
        best_fitness = max(final_configurations_fitnesses.keys())
        best_configuration = final_configurations_fitnesses[best_fitness]

        return best_configuration

    def start(self):
        for i in range(GENERATIONS):
            logger.info(f"Running generation {i+1}")
            self.population = self.evolve(self.population)

    def selection(
        self, parents: list, avg_evaluation: float, parents_evaluations: Iterable[float]
    ):
        if avg_evaluation:
            unnormalized_fitnesses = [pa / avg_evaluation for pa in parents_evaluations]
            sum_unnormalized_fitnesses = sum(unnormalized_fitnesses)

            fitnesses = [
                unnormalized_fitness / sum_unnormalized_fitnesses
                for unnormalized_fitness in unnormalized_fitnesses
            ]
        else:
            # If both parents are "unfit" return both
            fitnesses = [1 / len(parents)] * len(parents)

        return np.random.default_rng().choice(
            parents,
            p=fitnesses,
            size=2,
        )

    def evaluate(self, configuration) -> float:
        game = GameOfLife(configuration, self.grid_size)

        game.run()
        population_growth = game.max_population - game.starting_population

        if population_growth <= 0:
            return 0

        return population_growth / (self.grid_size**2)

    def initialize_population(self) -> Any:
        logger.info("Initializing population")
        population = []

        for _ in range(self.starting_population_size):
            population.append(self._initialize_individual())
        return population

    def _initialize_individual(self):
        grid = np.random.choice(
            2, size=(MAX_POPULATION_CLUSTERS, MAX_POPULATION_CLUSTERS)
        )

        return grid

    def evolve(self, population):
        new_population = []

        parents_evaluations = [self.evaluate(p) for p in population]
        avg_evaluation = np.average(parents_evaluations)

        self._avg_fitnesses.append(avg_evaluation)

        for _ in range(len(population) // 2):
            parent_1, parent_2 = self.selection(
                population, avg_evaluation, parents_evaluations
            )

            # Crossover
            child1 = self.crossover(parent_1, parent_2)
            child2 = self.crossover(parent_2, parent_1)

            # Mutation
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)

            # Add new individuals to the new population
            new_population.extend([child1, child2])

        return np.array(new_population)

    def mutation(self, individual):
        """
        Performs mutation on an individual with a certain mutation rate.
        """
        for i in range(MAX_POPULATION_CLUSTERS):
            for j in range(MAX_POPULATION_CLUSTERS):
                if np.random.rand() < self._mutation_rate:
                    individual[i, j] = 1 - individual[i, j]

        return individual

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        if np.random.rand() > self._crossover_rate:
            return parent1

        point = MAX_POPULATION_CLUSTERS // 2
        child = np.concatenate((parent1[:point], parent2[point:]))

        return child

    @property
    def avg_fitnesses(self):
        return self._avg_fitnesses
