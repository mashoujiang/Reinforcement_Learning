"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
"""

import sys

import numpy as np

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 80  # pixels
CENTRAL = UNIT / 2  # central of grids
SEAM = UNIT / 8  # seam width
INNER = CENTRAL - SEAM  # inner grid
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width
STATUS_SIZE = MAZE_H * MAZE_W  # status' shape


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = STATUS_SIZE
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()
        self.withdraw()
        self.position = self.get_init_position()

    def get_init_position(self):
        init_pos = np.zeros([MAZE_H, MAZE_W])
        init_pos[0][0] = 1
        return init_pos.reshape(-1)

    def get_position(self, s):
        init_pos = np.zeros([MAZE_H, MAZE_W])
        x = int(s[0] / UNIT)
        y = int(s[1] / UNIT)
        init_pos[x][y] = 1
        return init_pos.reshape(-1)

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([CENTRAL, CENTRAL])

        # hell
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - INNER, hell1_center[1] - INNER,
            hell1_center[0] + INNER, hell1_center[1] + INNER,
            fill='black')
        # hell
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - INNER, hell2_center[1] - INNER,
            hell2_center[0] + INNER, hell2_center[1] + INNER,
            fill='black')

        # create oval
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - INNER, oval_center[1] - INNER,
            oval_center[0] + INNER, oval_center[1] + INNER,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - INNER, origin[1] - INNER,
            origin[0] + INNER, origin[1] + INNER,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        self.canvas.delete(self.rect)
        origin = np.array([CENTRAL, CENTRAL])
        self.rect = self.canvas.create_rectangle(
            origin[0] - INNER, origin[1] - INNER,
            origin[0] + INNER, origin[1] + INNER,
            fill='red')
        # return observation
        s = self.canvas.coords(self.rect)
        return self.get_position(s)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move one_epoch

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 10
            done = True
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        return self.get_position(s_), reward, done

    def render(self):
        self.update()


def update():
    print("maze env's update()")
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break


if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()
