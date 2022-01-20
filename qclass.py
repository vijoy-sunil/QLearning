import numpy as np
from collections import defaultdict


class QClass:
    def __init__(self, grid_dim, obstacle_cells, goal_cell):
        # environment dimension
        # *-------------------------------------------------> Y
        # | (0, 0)  (0, 1)  (0, 2)  . .
        # |
        # | (1, 0)  (1, 1)  (1, 2)  . . .
        # |
        # | (2, 0)  (2, 1)  (2, 2)  . . . .
        # X
        self.dim_r, self.dim_c = grid_dim
        self.state_space = self.dim_r * self.dim_c

        # rewards/penalties
        self.goal_reward = 5
        self.obstacle_reward = -30
        # Negative rewards (i.e., punishments) are used for all states except the goal
        # This encourages the AI to identify the shortest path to the goal by minimizing
        # its punishments!
        self.empty_cell_reward = -1

        # create reward table
        self.rewards = np.full((self.dim_r, self.dim_c), self.empty_cell_reward)
        self.rewards[goal_cell[0], goal_cell[1]] = self.goal_reward
        for cell_x, cell_y in obstacle_cells:
            self.rewards[cell_x, cell_y] = self.obstacle_reward

        # numeric action codes: 0 = up, 1 = right, 2 = down, 3 = left
        self.actions = ['up', 'down', 'right', 'left']
        self.action_space = len(self.actions)

        # init q table to 0
        # we convert 2D state to 1D state, so the q table will be a 2d array
        # with dim_r * dim_c rows and action cols
        self.q_table = np.zeros((self.state_space, self.action_space))

    # convert 2d index to 1d index
    def get_index(self, row_index, column_index):
        return self.dim_c * row_index + column_index

    # display state space and action space
    def show_state_action_space(self):
        print('State Space: ', self.state_space)
        print('Action Space', self.action_space)

    # get random action
    def get_action_sample(self):
        return np.random.randint(self.action_space)

    # goal cell is the terminal state, when we stop an episode
    def is_terminal_state(self, row_index, column_index):
        if self.rewards[row_index, column_index] == self.goal_reward:
            return True
        else:
            return False

    # get a random, non-terminal starting location (sample state), terminal
    # states include goal cell, obstacle cell
    def get_state_sample(self):
        # get a random row and column index
        row_index = np.random.randint(self.dim_r)
        column_index = np.random.randint(self.dim_c)
        # continue choosing random row and column indexes until a non-terminal
        # state is identified
        while self.is_terminal_state(row_index, column_index):
            row_index = np.random.randint(self.dim_r)
            column_index = np.random.randint(self.dim_c)
        return row_index, column_index

    # display contents of rewards table
    def show_rewards_table(self):
        print('Rewards table')
        print(self.rewards)

    # display contents of q table
    def show_q_table(self):
        print('Q table')
        print(self.q_table)

    # get the next location based on the chosen action
    def step(self, row_index, column_index, action_index):
        new_row_index = row_index
        new_column_index = column_index

        if self.actions[action_index] == 'up' and row_index > 0:
            new_row_index -= 1
        elif self.actions[action_index] == 'right' and column_index < self.dim_c - 1:
            new_column_index += 1
        elif self.actions[action_index] == 'down' and row_index < self.dim_r - 1:
            new_row_index += 1
        elif self.actions[action_index] == 'left' and column_index > 0:
            new_column_index -= 1
        return new_row_index, new_column_index
