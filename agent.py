import tensorflow as tf 
import random 
import numpy as np 
from game import SnakeGame, Direction, Point 
from collections import deque 
from model import Linear_QNet, QTrainer
from utils2 import plot
MAX_MEMORY = 100_000
BATCH_SIZE = 200
LR = 0.001

class Agent:
    
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr = LR,gamma= self.gamma)
    
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x-20 , head.y)
        point_r = Point(head.x+20 , head.y)
        point_u = Point(head.x , head.y-20)
        point_d = Point(head.x , head.y+20)
         
        dir_l = game.direction == Direction.LEFT 
        dir_r = game.direction == Direction.RIGHT 
        dir_u = game.direction == Direction.UP 
        dir_d = game.direction == Direction.DOWN 
        state = [
            #straight
            (dir_l and game.is_collision( point_l)) or
            (dir_r and game.is_collision( point_r)) or
            (dir_u and game.is_collision( point_u)) or
            (dir_d and game.is_collision( point_d)),
            #right
            (dir_l and game.is_collision( point_u)) or
            (dir_r and game.is_collision( point_d)) or
            (dir_u and game.is_collision( point_r)) or
            (dir_d and game.is_collision( point_l)),
            #left
            (dir_l and game.is_collision( point_d)) or
            (dir_r and game.is_collision( point_u)) or
            (dir_u and game.is_collision( point_l)) or
            (dir_d and game.is_collision( point_r)),
            
            dir_l, 
            dir_r, 
            dir_u,
            dir_d,
            
            game.food.x < game.head.x,  #left
            game.food.x > game.head.x , #right
            game.food.y < game.head.y, #up 
            game.food.y > game.head.y #down
        ]
        return np.array(state, dtype= int)
        
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        #(()) makes it only have one tuple    

    def train_long_memory(self):
        if len(self.memory)> BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #list of tuples
        else:
            mini_sample = self.memory 
            
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done) 
        
         
    def get_action(self, state):
        if (self.n_games >=50):
            self.epsilon = 5
        else:
            self.epsilon = 55-self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 100) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = tf.convert_to_tensor(state, dtype = tf.float32)
            prediction = self.model(state0)
            move = tf.argmax(prediction, axis = 1).numpy().item()
            final_move[move] = 1
        return final_move

def train():
    plot_scores = [] 
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()
    while True:
        old_state = agent.get_state(game)
        
        final_move = agent.get_action(old_state)
        
        reward, done, score = game.play_step(final_move)
        new_state = agent.get_state(game)

        agent.train_short_memory(old_state, final_move, reward, new_state, done)

        agent.remember(old_state, final_move, reward, new_state, done)
        
        if done:
            
            game.reset() 
            agent.n_games +=1
            agent.train_long_memory()
            
            if score>record:
                record = score 
                agent.model.save()
                
            print("Game", agent.n_games, "Score", score, "Record", record)
            
            plot_scores.append(score)
            total_score +=score 
            mean_score = total_score /agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            
            
if __name__ == '__main__':
    train()

