
import random 

maze_grid = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

ROWS=len(maze_grid)
COLS=len(maze_grid[0])

START=(1,1)
GOAL=(9,13)

ACTIONS=[
    (-1,0), #up
    (1,0), #down
    (0,-1), #left
    (0,1) #right
]

def step(state, action):
    row, col = state
    new_row = row + ACTIONS[action][0]
    new_col = col + ACTIONS[action][1]

    if new_row < 0 or new_row >= ROWS or new_col < 0 or new_col >= COLS:
        return state, -10, False
    if maze_grid[new_row][new_col] == 0:
        return state, -10, False
    new_state = (new_row, new_col)
    if new_state == GOAL:
        return new_state, 100, True
    return new_state, -1, False



def make_q_table():
    q={}
    for row in range(0,len(maze_grid)):
        for col in range(0,len(maze_grid[0])):
            if maze_grid[row][col] != 0:
              q[(row,col)]=[0.0,0.0,0.0,0.0]
    return q

def choose_action(q_table, state,epsilon):
    roll_number=random.random()
    if roll_number<epsilon:
        return random.randint(0,3)
    else:
        return q_table[state].index(max(q_table[state]))


def update(q_table, state, action, reward, next_state, alpha, gamma):
      old_value = q_table[state][action]
      best_next = max(q_table[next_state])
      td_target = reward + gamma * best_next
      td_error = td_target - old_value
      q_table[state][action] = old_value + alpha * td_error
 
def train(episodes=2000, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,max_steps=500):

    q_table=make_q_table()

    for episode in range(episodes):
        state = START
        done = False
        steps =0 

        while not done and steps<max_steps:
            action = choose_action(q_table,state, epsilon)

            next_state, reward, done = step(state,action)

            update(q_table, state, action, reward,next_state, alpha ,gamma)

            state=next_state
            steps+=1
        epsilon = max(epsilon_min,epsilon*epsilon_decay)

    return q_table



def best_path(q_table, max_steps=500):
      state = START
      path = [state]
      for _ in range(max_steps):
          if state == GOAL:
              break
          # pick the BEST action (pure exploit, epsilon = 0)
          action = q_table[state].index(max(q_table[state]))
          next_state, reward, done = step(state, action)
          if next_state == state:
              break            # stuck against a wall, stop
          state = next_state
          path.append(state)
      return path



def print_path(path):
    path_set = set(path)
    for r in range(ROWS):
        line = ""
        for c in range(COLS):
            if (r, c) == START:      line += "S "
            elif (r, c) == GOAL:     line += "G "
            elif (r, c) in path_set: line += "* "   # learned route
            elif maze_grid[r][c] == 0: line += "# "  # wall
            else:                    line += ". "    # open, unused
        print(line)


def main():
    # This runs ONLY when the file is executed directly (python q_learning.py),
    # NOT when it is imported (from q_learning import *).
    q = train()
    path = best_path(q)
    print("Path length:", len(path), "steps")
    print("Reached goal:", path[-1] == GOAL)
    print_path(path)


if __name__ == "__main__":
    main()


