import numpy as np
import time
from random import *
from tkinter import *

from logic import *

SIZE = 500
GRID_LEN = 4
GRID_PADDING = 10

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"
BACKGROUND_COLOR_DICT = {2: "#eee4da", 4: "#ede0c8", 8: "#f2b179", 16: "#f59563", \
                         32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72", 256: "#edcc61", \
                         512: "#edc850", 1024: "#edc53f", 2048: "#edc22e"}
CELL_COLOR_DICT = {2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2", \
                   32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2", \
                   512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2"}
FONT = ("Verdana", 40, "bold")

KEY_UP = 0
KEY_DOWN = 1
KEY_LEFT = 2
KEY_RIGHT = 3

MAX_NUM = 2048

class GameGrid(Frame):
    def __init__(self):
        Frame.__init__(self)

        self.grid()
        self.master.title('2048')

        # self.gamelogic = gamelogic
        self.commands = {KEY_UP: up, KEY_DOWN: down, KEY_LEFT: left, KEY_RIGHT: right}
        self.n_actions = len(self.commands)
        self.n_features = GRID_LEN * GRID_LEN
        self.total_reward = 0.0
        self.max_num = MAX_NUM

        self.grid_cells = []
        self.init_grid()
        self.init_matrix()
        self.update_grid_cells()

    def init_grid(self):
        background = Frame(self, bg=BACKGROUND_COLOR_GAME, width=SIZE, height=SIZE)
        background.grid()
        for i in range(GRID_LEN):
            grid_row = []
            for j in range(GRID_LEN):
                cell = Frame(background, bg=BACKGROUND_COLOR_CELL_EMPTY, width=SIZE / GRID_LEN, height=SIZE / GRID_LEN)
                cell.grid(row=i, column=j, padx=GRID_PADDING, pady=GRID_PADDING)
                # font = Font(size=FONT_SIZE, family=FONT_FAMILY, weight=FONT_WEIGHT)
                t = Label(master=cell, text="", bg=BACKGROUND_COLOR_CELL_EMPTY, justify=CENTER, font=FONT, width=4,
                          height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def gen(self):
        return randint(0, GRID_LEN - 1)

    def init_matrix(self):
        self.matrix = new_game(4)

        self.matrix = add_two(self.matrix)
        self.matrix = add_two(self.matrix)

    def update_grid_cells(self):
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(text=str(new_number), bg=BACKGROUND_COLOR_DICT[new_number],
                                                    fg=CELL_COLOR_DICT[new_number])
        self.update_idletasks()

    def step(self, action):
        game_over = False
        self.matrix, done, reward = self.commands[action](self.matrix)
        if done:
            self.matrix = add_two(self.matrix)
            self.update_grid_cells()
            done = False
            if game_state(self.matrix) == 'win':
                self.grid_cells[1][1].configure(text="You", bg=BACKGROUND_COLOR_CELL_EMPTY)
                self.grid_cells[1][2].configure(text="Win!", bg=BACKGROUND_COLOR_CELL_EMPTY)
                game_over = True
            if game_state(self.matrix) == 'lose':
                self.grid_cells[1][1].configure(text="You", bg=BACKGROUND_COLOR_CELL_EMPTY)
                self.grid_cells[1][2].configure(text="Lose!", bg=BACKGROUND_COLOR_CELL_EMPTY)
                game_over = True
                reward = 0

        #self.total_reward += reward
        state = np.array(self.matrix).reshape(-1)

        return state, reward, game_over

    def reset(self):
        #self.init_grid()
        self.grid_cells[1][1].configure(text="You", bg=BACKGROUND_COLOR_CELL_EMPTY)
        self.grid_cells[1][2].configure(text="Lose!", bg=BACKGROUND_COLOR_CELL_EMPTY)
        self.update_grid_cells()
        self.init_matrix()
        self.update_grid_cells()
        #self.update_grid_cells()
        state = np.array(self.matrix).reshape(-1)
        self.total_reward = 0.0
        
        return state


    def generate_next(self):
        index = (self.gen(), self.gen())
        while self.matrix[index[0]][index[1]] != 0:
            index = (self.gen(), self.gen())
        self.matrix[index[0]][index[1]] = 2


if __name__ == "__main__":
    print("Welcome to play 2048!")
    gamegrid = GameGrid()
    # AI policy part
    commands = [KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT]
    for cmd in commands:
        s_, reward, done = gamegrid.step(cmd)
        print("reward={}".format(reward))

    print("*"*20)
    time.sleep(5)
    gamegrid.reset()
    time.sleep(5)
    for cmd in commands:
        s_, reward, done = gamegrid.step(cmd)
        print("reward={}".format(reward))

    gamegrid.grid_cells[1][1].configure(text="You", bg=BACKGROUND_COLOR_CELL_EMPTY)
    gamegrid.grid_cells[1][2].configure(text="Lose!", bg=BACKGROUND_COLOR_CELL_EMPTY)
    print("n_actions={},n_features={}".format(gamegrid.n_actions,gamegrid.n_features))
    gamegrid.mainloop()