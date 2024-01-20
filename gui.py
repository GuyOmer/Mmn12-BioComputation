import tkinter as tk
from typing import ClassVar


class GameOfLifeGUI:
    _CELL_SIZE: ClassVar = 10

    def __init__(self, game):
        self.game = game

        self.cell_size = type(self)._CELL_SIZE
        self.window = tk.Tk()
        self.window.title("Conway's Game of Life")
        self.canvas = tk.Canvas(
            self.window,
            width=self.game._grid_size * self.cell_size,
            height=self.game._grid_size * self.cell_size,
        )
        self.canvas.pack()

    def draw_grid(self):
        self.canvas.delete("all")

        for row in range(self.game.grid_size):
            for col in range(self.game.grid_size):
                if self.game._grid[row, col] == 1:
                    x1, y1 = col * self.cell_size, row * self.cell_size
                    x2, y2 = x1 + self.cell_size, y1 + self.cell_size

                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="black")

    def update(self):
        self.game.update_grid()
        self.draw_grid()
        self.window.after(1000, self.update)

    def run(self):
        self.update()
        self.window.focus()
        self.window.mainloop()
