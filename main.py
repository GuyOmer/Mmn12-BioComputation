import logging

import game_of_life
import gui
from methuselah_ga import MethuselahGA

GENERATIONS = 1000

# create logger with 'spam_application'
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(ch)


def main():
    ga = MethuselahGA(10, 50)
    best = ga.get_best_configuration()

    game = game_of_life.GameOfLife(best, 50)
    gui.GameOfLifeGUI(game).run()


if __name__ == "__main__":
    main()
