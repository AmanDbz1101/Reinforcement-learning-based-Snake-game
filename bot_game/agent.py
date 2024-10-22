import tensorflow as tf 
import numpy as np 
from game import SnakeGame, Direction, Point 
import tensorflow as tf 
from tensorflow.keras import layers, models


class Agent:
    
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
        
    
    def get_action(self, state, model):
        final_move = [0, 0, 0]

        state0 = tf.convert_to_tensor(state, dtype = tf.float32)
        prediction = model(state0)
        move = tf.argmax(prediction, axis = 1).numpy().item()
        final_move[move] = 1
        return final_move  

class CustomModel(tf.keras.Model):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomModel, self).__init__()
        self.dense1 = layers.Dense(hidden_size, activation='relu', input_shape=(input_size,))
        self.dense2 = layers.Dense(output_size, activation = 'softmax')

    def call(self, x):
        if len(x.shape) == 1:
            x = tf.expand_dims(x, axis = 0)
        x = self.dense1(x)
        return self.dense2(x)

def create_model(input, hidden, output):
    return CustomModel(input, hidden, output)
    
def train():
    record = 0
    agent = Agent()
    game = SnakeGame()
    
    while True:
        model = create_model(11, 256, 3)
        model.load_weights("../model/bot_model.weights.h5")
        old_state = agent.get_state(game)
        
        final_move = agent.get_action(old_state, model)
        
        done, score = game.play_step(final_move)

        
        if done:
            if score>record:
                record = score 
            game.reset()   
            total_score +=score 
            mean_score = total_score /agent.n_games
            print("Score: ", score, "Record: ", record, "Mean Score: ", mean_score)
            
            
if __name__ == '__main__':
    train()

