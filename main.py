import pygame
import sys
import time

from q_learning import *

# 1. Initialize Pygame and Setup Window
pygame.init()
initial_agent=pygame.image.load('agent.png')
# Grid Configuration
TILE_SIZE = 80  # Each maze square is 80x80 pixels
MAZE_ROWS = 11
MAZE_COLS = 15


class Agent:
    IMG=pygame.transform.scale(initial_agent,(80,80))
    def __init__(self,x,y):
        self.x=x
        self.y=y


SCREEN_WIDTH = MAZE_COLS * TILE_SIZE
SCREEN_HEIGHT = MAZE_ROWS * TILE_SIZE
ready = True
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("2D Maze Rendering")
clock = pygame.time.Clock()

# Colors (RGB)
COLOR_WALL = (40, 44, 52)      # Dark grey/black for walls
COLOR_PATH = (240, 240, 240)   # Light grey for open paths
COLOR_START = (46, 204, 113)   # Green for the start position
COLOR_GRID = (210, 210, 210)   # Subtle grid lines

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

agent_boy=Agent(0,0)


def draw_maze(surface, grid,agent_boy):
    global ready
    for row_idx, row in enumerate(grid):
        for col_idx, cell in enumerate(row):
          
            x = col_idx * TILE_SIZE
            y = row_idx * TILE_SIZE
  
            if cell == 0:
                color = COLOR_WALL
            elif cell == 1:
                color = COLOR_PATH
            elif cell == 2:
                color = COLOR_START
          
            rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(surface, color, rect)          
           
            if cell != 0:
                pygame.draw.rect(surface, COLOR_GRID, rect, 1)
    screen.blit(Agent.IMG,(agent_boy.x,agent_boy.y))   

def check(next_pos):
    col=int(next_pos[0]/TILE_SIZE)
    row=int(next_pos[1]/TILE_SIZE)
    col = max(0, min(col, MAZE_COLS - 1)) 
    row = max(0, min(row, MAZE_ROWS - 1))
    if(maze_grid[row][col]==0):
        return False
    else:
        return True
    
def move(agent_boy,event):
    temp=0
    if event == pygame.K_w:
        temp=agent_boy.y
        temp=temp-TILE_SIZE
        can_move=check((agent_boy.x,temp))
        if can_move:
            agent_boy.y=agent_boy.y-TILE_SIZE
    elif event == pygame.K_a:
        temp=agent_boy.x
        temp=temp-TILE_SIZE
        can_move=check((temp,agent_boy.y))
        if can_move:
          agent_boy.x=agent_boy.x-TILE_SIZE
    elif event == pygame.K_d:
        temp=agent_boy.x
        temp=temp+TILE_SIZE
        can_move=check((temp,agent_boy.y))
        if can_move:
          agent_boy.x=agent_boy.x+TILE_SIZE
    elif event == pygame.K_s:
        temp=agent_boy.y
        temp=temp+TILE_SIZE
        can_move=check((agent_boy.x,temp))
        if can_move:
          agent_boy.y= agent_boy.y + TILE_SIZE
    print(f"POS:{agent_boy.x},{agent_boy.y}")

key_MAP=[
   pygame.K_w,
     pygame.K_s,
   pygame.K_a,
   pygame.K_d
]


def state_to_pixels(state):
    row, col = state
    return col * TILE_SIZE, row * TILE_SIZE

font = pygame.font.SysFont(None, 28)
COLOR_GOAL = (231, 76, 60) 


# Training state — lives OUTSIDE the loop so it persists between frames.
# Each pass of the game loop advances training by exactly ONE step,

q_table = make_q_table()
episode = 0
epsilon = 1.0
state = START
steps = 0
done = False

EPISODES = 500
ALPHA, GAMMA = 0.1, 0.9
EPS_MIN, EPS_DECAY = 0.05, 0.995
MAX_STEPS = 500

agent_boy.x, agent_boy.y = state_to_pixels(START)

phase = "training"     
victory_state = START


def main():

    # Main Game Loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if phase == "training":
            if episode < EPISODES:
                # --- ONE training step per frame ---
                action = choose_action(q_table, state, epsilon)
                next_state, reward, done = step(state, action)
                update(q_table, state, action, reward, next_state, ALPHA, GAMMA)
                state = next_state
                steps += 1
                agent_boy.x, agent_boy.y = state_to_pixels(state)

                # episode finished? reset for the next one
                if done or steps >= MAX_STEPS:
                    episode += 1
                    epsilon = max(EPS_MIN, epsilon * EPS_DECAY)
                    state = START
                    steps = 0
                    done = False
            else:
                phase = "complete"          
                agent_boy.x, agent_boy.y = state_to_pixels(START)

        elif phase == "complete":
            if victory_state != GOAL:
                best = q_table[victory_state].index(max(q_table[victory_state]))
                ns, _, _ = step(victory_state, best)
                if ns != victory_state:
                    victory_state = ns
                    agent_boy.x, agent_boy.y = state_to_pixels(victory_state)

    
        screen.fill(COLOR_WALL)
        draw_maze(screen, maze_grid, agent_boy)

        # highlight the goal cell
        gx, gy = state_to_pixels(GOAL)
        pygame.draw.rect(screen, COLOR_GOAL, pygame.Rect(gx, gy, TILE_SIZE, TILE_SIZE), 4)

        
        if phase == "training":
            hud = f"Episode {episode}/{EPISODES}   epsilon={epsilon:.2f}"
        else:
            hud = "walking the best path (from Q-table)"
        screen.blit(font.render(hud, True, (255, 255, 0)), (10, 10))

        pygame.display.flip()
    
        clock.tick(500 if phase == "training" else 5)

    pygame.quit()
    sys.exit()

if __name__== "__main__":
    main()


