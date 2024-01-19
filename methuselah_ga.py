import logging
from typing import Any

import numpy as np

from game_of_life import GameOfLife

GENERATIONS = 100

logger = logging.getLogger(__name__)


class MethuselahGA:
    def __init__(self, population_size: int, board_size: int):
        self.population_size = population_size
        self.grid_size = board_size
        self.population = self.initialize_population()

        self._crossover_rate = 0.1
        self._mutation_rate = 0.06

    def get_best_configuration(self):
        self.start()

        best_configuration = max(
            self.calculate_fitness(individual) for individual in self.population
        )

        return best_configuration

    def start(self):
        for i in range(GENERATIONS):
            logger.info(f"Running generation {i+1}")
            self.evolve(self.population)

    def selection(self, parents: list):
        fitnesses = [self.calculate_fitness(p) for p in parents]
        return np.random.default_rng().choice(
            parents,
            p=[fitness / sum(fitnesses) for fitness in fitnesses],
            size=2,
        )

    def calculate_fitness(self, configuration) -> float:
        game = GameOfLife(configuration, self.grid_size)

        game.run()
        max_population = game.max_population

        return max_population / (self.grid_size**2)

    def initialize_population(self) -> Any:
        logger.info("Initializing population")
        return np.random.default_rng().integers(
            2,  # Yields 0 or 1
            size=(
                self.population_size,
                self.grid_size,
                self.grid_size,
            ),
        )

    def evolve(self, population):
        new_population = []
        for _ in range(len(population) // 2):
            parent_1, parent_2 = self.selection(population)

            # Crossover
            logger.debug("Crossover")
            child1 = self.crossover(parent_1, parent_2)
            child2 = self.crossover(parent_2, parent_1)

            # Mutation
            logger.debug("Mutation")
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)

            # Add new individuals to the new population
            new_population.extend([child1, child2])

        return np.array(new_population)

    def mutation(self, individual):
        """
        Performs mutation on an individual with a certain mutation rate.
        """
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if np.random.rand() < self._mutation_rate:
                    individual[i, j] = 1 - individual[i, j]
                    logger.debug("Mutation operation performed.")

        return individual

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        if np.random.rand() > self._crossover_rate:
            return parent1

        point = np.random.randint(1, self.grid_size - 1)
        child = np.concatenate((parent1[:point], parent2[point:]))
        logger.debug("Crossover operation performed.")

        return child
