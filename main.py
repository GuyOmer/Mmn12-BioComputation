import logging

import matplotlib.pyplot as plt
import pandas as pd

import game_of_life
import gui
from constants import GRID_SIZE, CROSSOVER_RATE, MUTATION_RATE, STARTING_POPULATION_SIZE
from methuselah_ga import MethuselahGA

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def main():
    ga = MethuselahGA(STARTING_POPULATION_SIZE, GRID_SIZE)
    best = ga.get_best_configuration()

    game = game_of_life.GameOfLife(best, GRID_SIZE)
    gui.GameOfLifeGUI(game).run()

    df = pd.DataFrame(
        {
            "Round": range(1, len(ga.avg_fitnesses) + 1),
            "Average Fitness": ga.avg_fitnesses,
        }
    )
    df.plot(kind="scatter", x="Round", y="Average Fitness", color="blue", alpha=0.5)
    plt.title(f"Average Fitness\nCrossover {CROSSOVER_RATE} Mutation {MUTATION_RATE}")

    plt.xlabel("Round")
    plt.ylabel("Average Fitness")
    plt.grid(True)

    df.to_excel(f"crossover {CROSSOVER_RATE} mutation {MUTATION_RATE}.xlsx")

    # Showing the plot
    plt.show()
    logger.info(f"Max size: {game.max_population}")


if __name__ == "__main__":
    main()
