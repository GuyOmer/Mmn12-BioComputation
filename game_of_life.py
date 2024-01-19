import collections
import logging

import numpy as np

logger = logging.getLogger(__name__)

MAX_ROUNDS = 1000


class GameOfLife:
    def __init__(self, starting_configuration, grid_size: int):
        self._grid_size = grid_size
        self._grid = starting_configuration

        self._max_population = 0

        self._history = collections.deque(maxlen=20)

    def run(self):
        for i in range(MAX_ROUNDS):
            # logger.debug(f'Round {i+1}')
            self.update_grid()

            current_hash = self.hash_grid()
            if current_hash in self._history:
                break

            self._history.append(self.hash_grid())

            self._max_population = max(
                self._max_population,
                self._count_living_cells(),
            )

    def update_grid(self):
        new_grid = self._grid.copy()

        for row in range(self._grid_size):
            for col in range(self._grid_size):
                live_neighbors = self._count_live_neighbors(row, col, self._grid)

                if self._grid[row, col] == 1:
                    if live_neighbors < 2 or live_neighbors > 3:
                        new_grid[row, col] = 0
                elif live_neighbors == 3:
                    new_grid[row, col] = 1

        self._grid = new_grid

    def hash_grid(self) -> int:
        live_cells = set()
        first_scanned_cell = None

        for row in range(self._grid_size):
            for col in range(self._grid_size):
                if self._grid[row, col]:
                    if first_scanned_cell is None:
                        first_scanned_cell = (row, col)

                    live_cells.add(
                        (
                            row - first_scanned_cell[0],
                            col - first_scanned_cell[1],
                        )
                    )

        return hash(frozenset(live_cells))

    def _count_living_cells(self) -> int:
        count = np.sum(self._grid)

        return count

    def _count_live_neighbors(self, row: int, col: int, grid) -> int:
        count = (
            np.sum(
                grid[
                    max(0, row - 1) : min(row + 2, self.grid_size),
                    max(0, col - 1) : min(col + 2, self.grid_size),
                ]
            )
            - grid[row, col]
        )

        return count

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @property
    def max_population(self) -> int:
        return self._max_population
