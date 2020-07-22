import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import numpy as np
from collections import deque
import random
import copy

class DQN():

    def __init__(self, env, epsilon, epsilon_decay, epsilon_cutoff, alpha, gamma, numEpisodes, stepSize, batchSize, weight_file, memorySize, startMemSize, numMaxStep=1000, numEpoch=1):

        self.env = env
        self.numActions = self.env.action_space.n
        self.numStateVar = self.env.observation_space.shape[0]

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_cutoff = epsilon_cutoff
        self.alpha = alpha
        self.gamma = gamma
        self.numEpisodes = numEpisodes
        self.weight_file = weight_file

        self.batchSize = batchSize
        self.memory = deque(maxlen=memorySize)
        self.startMemSize = startMemSize
        self.numMaxStep = numMaxStep
        self.numEpoch = numEpoch
        self.stepSize = stepSize

        self.Q = None
        self.QHat = None
        self.step = 0
        self.training_losses = np.empty(0)
        self.training_rewards = np.empty(0)

    # this funciton creates an ANN based on the input dimensions
    # it also adds the first layer which corresponds to the number of state variables
    # and the last layer which corresponds to the number of actions
    # you can also load pretrained weights from self.weight_file.h5
    def createANNModel(self, dimensions):
        tf.compat.v1.disable_eager_execution()
        self.Q = Sequential()
        self.Q.add(Dense(dimensions[0], input_shape=(self.numStateVar,), activation='relu'))
        for i in range(1,len(dimensions)):
            self.Q.add(Dense(dimensions[i], activation='relu'))
        self.Q.add(Dense(self.numActions))
        # clipnorm, clipvalue
        opt = optimizers.Adam(learning_rate=self.alpha)
        self.Q.compile(loss='mse', optimizer=opt)
        try:
            self.Q.load_weights(self.weight_file+".h5")
            print("*********** weights loaded ***********")
        except:
            print("*********** no weights loaded ***********")
        self.Q.summary()

        # according to the DQN paper, set QHat aside for estimating new target for training Q
        # updating QHat every ${self.stepSize} steps
        self.QHat = clone_model(self.Q)
        self.QHat.set_weights(self.Q.get_weights())

    # choose action using self.Q
    def __chooseAction(self, epsilon_adj, state):
        action = None
        # explore
        if np.random.random() < epsilon_adj:
            action = self.env.action_space.sample()
        # exploit
        else:
            action = np.argmax(self.Q.predict(
                np.reshape(state, (1, self.numStateVar)))[0])
        return action

    # train DQN once with memory replay
    def __trainModel(self):
        # creating training set from random sampling
        trainingBatch = random.sample(self.memory, self.batchSize)
        X_train = np.empty(shape=(0, self.numStateVar), dtype=float)
        Y_train = np.empty(shape=(0, self.numActions), dtype=float)

        states = np.array([data[0] for data in trainingBatch])
        s_primes = np.array([data[3] for data in trainingBatch])
        # this is the current Q values
        Q_s_all = self.Q.predict(states)
        # QHat is only used here for calculating target Q values
        Q_prime_all = self.QHat.predict(s_primes)

        for i in range(self.batchSize):
            state = np.reshape(trainingBatch[i][0], (1, self.numStateVar))
            action = trainingBatch[i][1]
            reward = trainingBatch[i][2]
            done = trainingBatch[i][4]
            Q_s = copy.copy(Q_s_all[i])
            # if not terminal state Q(s,a) = r + gamma * max Q(s_prime,a)
            if not done:
                maxQprime = np.max(Q_prime_all[i])
                Q_s[action] = reward + self.gamma * maxQprime
            # if terminal state, update Q(s,a) = r
            else:
                Q_s[action] = reward
            Q_s = np.reshape(Q_s, (1, self.numActions))
            # add sample to training set
            X_train = np.append(X_train, state, axis=0)
            Y_train = np.append(Y_train, Q_s, axis=0)
        result = self.Q.fit(X_train, Y_train,
                            batch_size=self.batchSize,
                            epochs=self.numEpoch, verbose=0)
        self.training_losses = np.append(
            self.training_losses, result.history["loss"][-1])

    # run the enviroment once using DQN
    # for testing DQN, set eplison_adj to 0 (no exploration)
    def runOneEpisode(self, epsilon_adj, render, train):
        totalReward = 0
        state = self.env.reset()
        for i in range(self.numMaxStep):
            action = self.__chooseAction(epsilon_adj, state)
            s_prime, reward, done, _ = self.env.step(action)
            self.memory.append((state, action, reward, s_prime, done))
            if render:
                self.env.render()
            if train:
                self.__trainModel()
                # update model every ${self.stepSize} steps
                self.step += 1
                if self.step >= self.stepSize:
                    self.QHat.set_weights(self.Q.get_weights())
                    self.step = 0
            state = s_prime
            totalReward += reward
            if done:
                break
        #print("step: {} \r".format(i), end="")
        return totalReward

    def trainDQN(self, interval):
        self.step = 0
        epsilon_adj = self.epsilon
        # first populate the memmory with complete random actions without training 
        while len(self.memory)<self.startMemSize:
          self.runOneEpisode(1, render=False, train=False)
        # train model
        for i in range(self.numEpisodes):
            r = self.runOneEpisode(epsilon_adj, render=False, train=True)
            self.training_rewards = np.append(self.training_rewards, r)
            
            if i % interval == 0 and i > 0:
                # uncomment code below to save ANN weights and performance data along training
                # self.Q.save_weights("%s_%d.h5" % (self.weight_file, i))
                # np.save("training_errors", self.training_losses)
                # np.save("training_rewards", self.training_rewards)
                print("\nepisode {}: loss-->{:.2f}+/-{:.2f}, reward-->{:.2f}+/-{:.2f}".format(i, 
                    np.average(self.training_losses[-interval:-1]),
                    np.std(self.training_losses[-interval:-1]),  
                    np.average(self.training_rewards[-interval:-1]),
                    np.std(self.training_rewards[-interval:-1])))
                
            if epsilon_adj > self.epsilon_cutoff:    
              epsilon_adj *= self.epsilon_decay

    def evaluateDQN(self, numRuns):
        rewards = np.empty(0)
        for i in range(numRuns):
            rewards = np.append(rewards, self.runOneEpisode(
                0, render=False, train=False))
        return rewards
