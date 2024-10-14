import tensorflow as tf
from tensorflow.keras import layers
import os
import numpy as np

class Linear_QNet(tf.keras.Model):
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet, self).__init__()
        self.dense1 = layers.Dense(hidden_size, activation='relu', input_shape=(input_size,))
        self.dense2 = layers.Dense(output_size)

    def call(self, x):
        if len(x.shape) == 1:
            x = tf.expand_dims(x, axis = 0)
        x = self.dense1(x)
        return self.dense2(x)

    def save(self, file_name='new_model.weights.h5'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_path = os.path.join(model_folder_path, file_name)
        self.save_weights(file_path)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def train_step(self, state, action, reward, next_state, done):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        # Reshape the tensors if necessary
        if len(state.shape) == 1:
            state = np.expand_dims(state, 0)
            next_state = np.expand_dims(next_state, 0)
            action = np.expand_dims(action, 0)
            reward = np.expand_dims(reward, 0)
            done = (done,)

        # Forward pass (prediction)
        with tf.GradientTape() as tape:
            pred = self.model(state)
            # target = tf.identity(pred)
            target = pred.numpy()
            i=0
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    Q_new = reward[idx] + self.gamma * tf.reduce_max(self.model(next_state[idx]))
                index = int(np.argmax(action))
                target[idx][index] = Q_new

            # Compute loss
            loss = self.loss_fn(target, pred)

        # Backward pass (gradient update)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        