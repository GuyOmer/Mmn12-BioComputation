import collections
import logging

import numpy as np
import xxhash
from scipy.signal import convolve2d

logger = logging.getLogger(__name__)

MAX_ROUNDS = 1000

NEIGHBORS_KERNEL = np.array(
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]
)


class GameOfLife:
    def __init__(self, starting_configuration: np.array, grid_size: int):
        self._grid_size = grid_size

        grid = np.zeros((self.grid_size, self.grid_size))

        # Calculate the start and end indices to place the population cluster in the center of the grid
        start_index_x = (grid.shape[1] - starting_configuration.shape[1]) // 2
        start_index_y = (grid.shape[0] - starting_configuration.shape[0]) // 2
        end_index_x = start_index_x + starting_configuration.shape[1]
        end_index_y = start_index_y + starting_configuration.shape[0]

        # Place the small array in the center of the large array
        grid[
            start_index_y:end_index_y, start_index_x:end_index_x
        ] = starting_configuration
        self._grid = grid

        self._starting_population = self._count_living_cells()

        self._max_population = 0

        self._history = collections.deque(maxlen=20)

    def run(self):
        for i in range(MAX_ROUNDS):
            self.update_grid()

            current_hash = self.hash_grid()
            if current_hash in self._history:
                logger.debug(f"Stopping at round {i+1}")
                break

            self._history.append(self.hash_grid())
        logger.debug(f"Stopping at round {MAX_ROUNDS}")

    def update_grid(self):
        # Get the count of live neighbors for each cell
        neighbors = self.count_live_neighbors_convolution(self._grid)

        # Cells that will die (live cell with fewer than 2 or more than 3 neighbors)
        death_mask = (self._grid == 1) & ((neighbors < 2) | (neighbors > 3))

        # Cells that will come to life (dead cell with exactly 3 neighbors)
        birth_mask = (self._grid == 0) & (neighbors == 3)

        self._grid[death_mask] = 0
        self._grid[birth_mask] = 1

        self._max_population = max(
            self._max_population,
            self._count_living_cells(),
        )

    def count_live_neighbors_convolution(self, grid):
        grid_size = grid.shape[0]
        # Apply convolution and use 'same' mode to keep the output size equal to the input size
        neighbor_count = convolve2d(
            grid, NEIGHBORS_KERNEL, mode="same", boundary="fill", fillvalue=0
        )
        return neighbor_count  # This will be a 2D array with the count of live neighbors for each cell

    def hash_grid(self) -> str:
        array_bytes = self._grid.ravel().view(np.uint8).tobytes()

        # Create a hash object and return the hash
        return xxhash.xxh64(array_bytes).hexdigest()

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

    @property
    def starting_population(self):
        return self._starting_population
