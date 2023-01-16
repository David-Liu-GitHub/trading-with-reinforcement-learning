import numpy as np
import pandas as pd
import math
import sys
import os
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
import random
from collections import deque
import yfinance as yf


class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size
        self.action_size = 3  # hold, buy, and sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = load_model("models/" + model_name) if is_eval else self._model()

    def _model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))
        return model

    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        options = self.model.predict(state)
        return np.argmax(options[0])

    def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# prints formatted price
def formatPrice(n):
    if n < 0:
        return "-$" + "{0:.2f}".format(abs(n))
    else:
        return "$" + "{0:.2f}".format(abs(n))


# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
    historyData = yf.download(key, period="30d", interval="1d")
    closePrice = historyData["Close"].tolist()
    return closePrice


# returns the sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# returns an n-day state ending at time t
def getState(data, endDate, windowSize):
    startDate = endDate - windowSize + 1
    if startDate >= 0:
        block = data[startDate:(endDate + 1)]
    else:
        block = -startDate * [data[0]] + data[0:(endDate + 1)]  # If not enough data to populate one window, repeat the
                                                                # first close price to populate the window state
    priceChange = []
    for i in range(windowSize - 1):
        priceChange.append(sigmoid(block[i + 1] - block[i]))
    return np.array([priceChange])


# Train and save the model
def trainModel(stock_name, window_size, epoch_count):
    agent = Agent(window_size)
    data = getStockDataVec(stock_name)
    l = len(data) - 1
    batch_size = 32
    for epoch in range(epoch_count + 1):
        print("Epoch " + str(epoch) + "/" + str(epoch_count))
        state = getState(data, 0, window_size + 1)
        total_profit = 0
        agent.inventory = []
        for t in range(l):
            action = agent.act(state)
            next_state = getState(data, t + 1, window_size + 1)
            reward = 0
            # Hold
            if action == 0:
                print("Hold")
            # Buy
            if action == 1:
                agent.inventory.append(data[t])
                print("Buy: " + formatPrice(data[t]))
            # Sell
            elif action == 2 and len(agent.inventory) > 0:
                bought_price = agent.inventory.pop(0)
                reward = max(data[t] - bought_price, 0)
                total_profit += data[t] - bought_price
                print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
            # Indicate if done
            done = False
            if t == l - 1:
                done = True
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                print("--------------------------------")
                print("Total Profit: " + formatPrice(total_profit))
                print("--------------------------------")
            if len(agent.memory) > batch_size:
                agent.expReplay(batch_size)

        if epoch % 10 == 0:  # Saves model every 10 episodes
            agent.model.save("/working/model_ep" + str(epoch))

        return agent


if __name__ == "__main__":
    stock_name, window_size, episode_count = '^GSPC', 12, 5
    agent = trainModel(stock_name, window_size, episode_count)







