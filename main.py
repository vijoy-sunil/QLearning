from qclass import QClass
import numpy as np

# Environment parameters
grid_dims = [3, 3]
obstacle_cells = [[0, 1], [1, 1]]
goal_cell = [0, 2]
num_episodes = 10000
inference_start = [0, 0]

# Hyperparameters
# alpha is the learning rate, gamma is the discount factor
# applied on the potential reward in the future and epsilon decides
# exploration - exploitation ratio
alpha = 0.1
gamma = 0.6
epsilon = 0.2


def main():
    QL = QClass(grid_dims, obstacle_cells, goal_cell)
    QL.show_state_action_space()

    QL.show_rewards_table()

    for i in range(0, num_episodes):
        # get starting cell (random sample)
        row_index, column_index = QL.get_state_sample()
        print('Initial state', row_index, column_index)

        # epochs for completion of 1 iteration in an episode
        epochs = 0
        # keep track of rewards
        rewards = 0

        # continue taking actions (i.e., moving) until we reach a terminal state
        while not QL.is_terminal_state(row_index, column_index):
            # exploration
            if np.random.uniform(0, 1) < epsilon:
                action = QL.get_action_sample()
            # exploitation
            else:
                action = np.argmax(QL.q_table[QL.get_index(row_index, column_index)])

            # perform action
            old_row_index, old_column_index = row_index, column_index
            row_index, column_index = QL.step(old_row_index, old_column_index, action)

            # observation
            reward = QL.rewards[row_index, column_index]
            # cumulative reward through this episode
            rewards += reward

            old_q_value = QL.q_table[QL.get_index(old_row_index, old_column_index), action]
            target_q_value = reward + (gamma * np.max(QL.q_table[QL.get_index(row_index, column_index)]))

            # update q table entry for current state
            QL.q_table[QL.get_index(old_row_index, old_column_index), action] = \
                old_q_value + (alpha * (target_q_value - old_q_value))

            # update epoch
            epochs += 1

        print("Episode count: %d, epochs: %d, rewards: %d" % (i, epochs, rewards))

    print('Training complete')
    QL.show_q_table()

    # inference
    # find the shortest path to goal cell using the trained q table
    path = []
    if QL.is_terminal_state(inference_start[0], inference_start[1]):
        return path
    # it is a legal start cell
    else:
        path.append(inference_start)
        current_row_index, current_column_index = inference_start
        # continue until we reach the goal
        while not QL.is_terminal_state(current_row_index, current_column_index):
            # get the best action from q table
            action = np.argmax(QL.q_table[QL.get_index(current_row_index, current_column_index)])
            # take the action
            current_row_index, current_column_index = QL.step(current_row_index, current_column_index, action)
            # append to path
            path.append([current_row_index, current_column_index])

    print("Shortest path from {} to {}" .format(inference_start, goal_cell))
    print(path)


if __name__ == "__main__":
    main()
