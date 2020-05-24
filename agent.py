import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import random
import os
import pygame

import numpy as np
import matplotlib.pyplot as plt

from operator import add
from collections import deque

from environment import game_ai


#NETWORK: input size = 11, hidden size = 256, output size = 3
class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class DQNAgent(object):
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0
        self.counter_games = 0
        #replay memory 
        self.memory = deque() 
        self.model = QNet(11, 256, 3)
        self.model.train()

        #https://arxiv.org/abs/1412.6980 Adam: A Method for Stochastic Optimization
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.loss_fn = nn.MSELoss()

    def get_state(self, snake):
        state = [
            # immediate danger for snake straight, right, or left
            (snake.x_change == 20 and snake.y_change == 0 and ((list(map(add, snake.snakeSegments[0], [20, 0])) in snake.snakeSegments) or snake.snakeSegments[0][0] + 20 >= (snake.display_width - 20))) or 
            (snake.x_change == -20 and snake.y_change == 0 and ((list(map(add, snake.snakeSegments[0], [-20, 0])) in snake.snakeSegments) or snake.snakeSegments[0][0] - 20 < 20)) or 
            (snake.x_change == 0 and snake.y_change == -20 and ((list(map(add, snake.snakeSegments[0], [0, -20])) in snake.snakeSegments) or snake.snakeSegments[0][-1] - 20 < 20)) or 
            (snake.x_change == 0 and snake.y_change == 20 and ((list(map(add, snake.snakeSegments[0], [0, 20])) in snake.snakeSegments) or snake.snakeSegments[0][-1] + 20 >= (snake.display_height-20))),

            (snake.x_change == 0 and snake.y_change == -20 and ((list(map(add,snake.snakeSegments[0],[20, 0])) in snake.snakeSegments) or snake.snakeSegments[0][0] + 20 > (snake.display_width-20))) or 
            (snake.x_change == 0 and snake.y_change == 20 and ((list(map(add,snake.snakeSegments[0],[-20,0])) in snake.snakeSegments) or snake.snakeSegments[0][0] - 20 < 20)) or 
            (snake.x_change == -20 and snake.y_change == 0 and ((list(map(add,snake.snakeSegments[0],[0,-20])) in snake.snakeSegments) or snake.snakeSegments[0][-1] - 20 < 20)) or 
            (snake.x_change == 20 and snake.y_change == 0 and ((list(map(add,snake.snakeSegments[0],[0,20])) in snake.snakeSegments) or snake.snakeSegments[0][-1] + 20 >= (snake.display_height-20))),

            (snake.x_change == 0 and snake.y_change == 20 and ((list(map(add,snake.snakeSegments[0],[20,0])) in snake.snakeSegments) or snake.snakeSegments[0][0] + 20 > (snake.display_width-20))) or 
            (snake.x_change == 0 and snake.y_change == -20 and ((list(map(add, snake.snakeSegments[0],[-20,0])) in snake.snakeSegments) or snake.snakeSegments[0][0] - 20 < 20)) or 
            (snake.x_change == 20 and snake.y_change == 0 and ((list(map(add,snake.snakeSegments[0],[0,-20])) in snake.snakeSegments) or snake.snakeSegments[0][-1] - 20 < 20)) or 
            (snake.x_change == -20 and snake.y_change == 0 and ((list(map(add,snake.snakeSegments[0],[0,20])) in snake.snakeSegments) or snake.snakeSegments[0][-1] + 20 >= (snake.display_height-20))),

            # direction snake is currently moving
            snake.x_change == -20, 
            snake.x_change == 20,  
            snake.y_change == -20,  
            snake.y_change == 20,

            # fruit location 
            snake.fruitPosition[0] < snake.snakePosition[0],  # fruit left
            snake.fruitPosition[0] > snake.snakePosition[0],  # fruit right
            snake.fruitPosition[1] < snake.snakePosition[1],  # fruit up
            snake.fruitPosition[1] > snake.snakePosition[1]  # fruit down
            ]

        #change from boolean to integer vals
        for i in range(len(state)):
            if state[i]:
                state[i]=1
            else:
                state[i]=0


        #Convert state to an ndarray
        return np.asarray(state)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])
        if len(self.memory) > 100000:
            self.memory.popleft()

    def replay_memory(self, memory):
        self.counter_games += 1
        if len(memory) > 1000:
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory
        
        state, action, reward, next_state, done = zip(*minibatch)
        state = torch.tensor(state, dtype=torch.float) #[1, ... , 0]
        action = torch.tensor(action, dtype=torch.long) # [1, 0, 0]
        reward = torch.tensor(reward, dtype=torch.float) # int
        next_state = torch.tensor(next_state, dtype=torch.float) #[True, ... , False]

        target = reward
        if not done:
            target = reward + self.gamma * torch.max(self.model(next_state))

        #optimize
        location = [[x] for x in torch.argmax(action, dim=1).numpy()]
        location = torch.tensor(location)
        prediction = self.model(state).gather(1, location)#[action]
        prediction = prediction.squeeze(1)
        loss = self.loss_fn(target, prediction)
        loss.backward()
        self.optimizer.step()

    def train_on_new_state(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        target = reward
        if not done:
            target = reward + self.gamma * torch.max(self.model(next_state))

        #optimize
        prediction = self.model(state)
        target_f = prediction.clone()
        target_f[torch.argmax(action).item()] = target
        loss = self.loss_fn(target_f, prediction)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def plot(self, score, mean_per_game):
        from IPython import display
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()
        plt.title('Training Results')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(score)
        plt.plot(mean_per_game)
        plt.ylim(ymin=0)
        plt.text(len(score)-1, score[-1], str(score[-1]))
        plt.text(len(mean_per_game)-1, mean_per_game[-1], str(mean_per_game[-1]))
    
    def get_action(self, state):
        #after 80 games the agent only uses the policy to make decisions
        self.epsilon = 80 - self.counter_games
        final_move = [0, 0, 0]
        #off-policy; randomly chooses movements
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] += 1
        #on-policy; uses network to choose action
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] += 1
        return final_move



def train():
    #will save the model for the game that gets the highest score
    save_model = False

    #have to set window caption in here lol idk why
    pygame.display.set_caption('SmartSnake')

    #make folder for saved models
    if save_model:
        if not os.path.exists('./model'):
            os.makedirs('./model')

    #setup for training

    #Turn the interactive mode on for pyplot
    plt.ion()

    #Initialize plotting values
    score_plot = []
    total_score = 0
    mean_plot =[]
    record = 0

    #initialize agent and environment
    agent = DQNAgent() #agent == snake
    game = game_ai() #enviroment == game/board/emulator

    #max number of games the agent will play while training
    max_games = 200

    while True:
        if agent.counter_games > max_games:
            avg = total_score / agent.counter_games
            exit_str = "Max number of games reached\nRecord: "+str(record)+" Avg: "+str(avg)
            exit(exit_str)

        #get old state
        state_old = agent.get_state(game)
        
        final_move = agent.get_action(state_old)

        #perform new move and get new state
        reward, done, score = game.frameStep(final_move)
        state_new = agent.get_state(game)
    
        #train model given new state
        agent.train_on_new_state(state_old, final_move, reward, state_new, done)
        
        # store the new data into a long term memory
        agent.remember(state_old, final_move, reward, state_new, done)

        if done == True:
            # once game is over, train on the memory and plot the result
            sc = game.reset()
            total_score += sc
            agent.replay_memory(agent.memory)

            #print what number game the snake is on and what the score is for the game
            print('Game', agent.counter_games, '      Score:', sc)

            #update record and save model if new high score 
            #model from iteration w highest score saved as best_model.pth
            if sc > record:
                record = sc
                if save_model:
                    dir = os.path.join('./model', 'best_model.pth')
                    torch.save(agent.model.state_dict(), dir)

            print('record: ', record)
            #add plot points for current game: score and updated average
            score_plot.append(sc)
            mean = total_score / agent.counter_games
            mean_plot.append(mean)
            agent.plot(score_plot, mean_plot)

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    #load game icon and initialize pygame
    image = pygame.image.load('snake.png')
    pygame.display.set_icon(image)
    pygame.init()

    #call training function!!!!
    train()
