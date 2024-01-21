import torch
import random
import numpy as np
from snake_game import SnakeGameAI, Direction, Point
from collections import deque #A data structure to store memory !reread
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000 #Store 100000 items in memory
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #control the randomness
        self.gamma = 0.8 # discount rate in the deep Q learning 
        self.memory = deque(maxlen=MAX_MEMORY) #If we exceed the memory, we pop left
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma = self.gamma)
        # model, trainer


    def get_state(self, game):
        head = game.snake[0]
        #Moving point of snake (1 move ahead)
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        #Check direction
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        #11 state as we mentioned
        #1. Danger 3 , 2. Direction X 4, Food x 4
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_r and game.is_collision(point_d)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)),

            # Danger left
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_d and game.is_collision(point_r)),

            #Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,
        ]

        return np.array(state, dtype=int) #Save as np to integer
        

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #pop to left if MAX_MEMORY exceeds (()) add a tuple

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # Return a list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample) #Reread the zip function
        self.trainer.train_step(states, actions, rewards, next_states, dones)#Can handle multiple sizes


    def train_short_memory(self,  state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        #Random moves: tradeoff exploration / exploitation
        #The better our agent gets, the less random we want to make
        self.epsilon = 80 - self.n_games#hardcore 80 cell
        final_move = [0, 0, 0]
        #Random the move
        if random.randint(0, 200) < self.epsilon: #If the epsilon small enough, we dont want to random
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float) #reread
            prediction = self.model(state0)
            move = torch.argmax(prediction).item() #Argmax
            final_move[move] = 1

        return final_move

def train():
    plot_scores = [] #saves for plotting
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # Get old state
        state_old = agent.get_state(game) #get the game

        # Get move
        final_move = agent.get_action(state_old)

        #perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        #train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        #remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, "Record:", record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
if __name__ == '__main__':
    train()
