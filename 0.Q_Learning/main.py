# Let's build Q-Learning stepby step from the first prinicples without using any libraries.

import random

GRID_SIZE = 5
ACTIONS = ['up', 'right', 'down', 'left']
MOVE = {'up': (0,-1), 'right': (1,0), 'down': (0,1), 'left': (-1,0)}

def create_q_table() :
    return {(x,y): [0] * len(ACTIONS) for x in range(GRID_SIZE) for y in range(GRID_SIZE)}  

def q_learn(alpha,gamma,epsilon,episodes):
    q_table = create_q_table()
    for ep in range(episodes):
        x,y,steps = 0,0,0
        while steps < 200:
            steps += 1
            key = (x,y)
            if random.random() < epsilon:
                action_idx = random.randint(0 , 3)
            else:
                action_idx = q_table[key].index(max(q_table[key]))

            dx,dy = MOVE[ACTIONS[action_idx]]
            nx = max(0, min(GRID_SIZE-1, x + dx))
            ny = max(0, min(GRID_SIZE-1, y + dy))
            done = nx == GRID_SIZE-1 and ny == GRID_SIZE-1
            reward = 10 if done else -0.1
            max_next_q = max(q_table[(nx,ny)])
            q_table[key][action_idx] += alpha * (reward + gamma * max_next_q - q_table[key][action_idx])
            x, y = nx, ny
            if done:
                break
    return q_table

q_table = q_learn(0.1,0.9,0.3,500)
print("Training complete!.............")
print("Starting point (0,0) Q-values:", [f'{v:.2f}' for v in q_table[(0,0)]])                
