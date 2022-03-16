import random
from typing import Optional
from numpy import ndarray
import numpy as np
from model import QModel
import torch
import random as rd

class DQNAgent:
    """
    The agent class for exercise 1.
    """

    def __init__(self,
                 obs_dim: int,
                 num_actions: int,
                 learning_rate: float,
                 gamma: float,
                 epsilon_max: Optional[float] = None,
                 epsilon_min: Optional[float] = None,
                 epsilon_decay: Optional[float] = None):
        """
        :param num_states: Number of states.
        :param num_actions: Number of actions.
        :param learning_rate: The learning rate.
        :param gamma: The discount factor.
        :param epsilon_max: The maximum epsilon of epsilon-greedy.
        :param epsilon_min: The minimum epsilon of epsilon-greedy.
        :param epsilon_decay: The decay factor of epsilon-greedy.
        """
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.nn = QModel(obs_dim, num_actions)
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon = epsilon_max
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=self.learning_rate)

    def greedy_action(self, observation) -> int:
        """
        Return the greedy action.

        :param observation: The observation.
        :return: The action.
        """
        action = torch.argmax(self.nn(observation))
        return action

    def act(self, observation, training: bool = True) -> int:
        """
        Return the action.

        :param observation: The observation.
        :param training: Boolean flag for training, when not training agent
        should act greedily.
        :return: The action.
        """
        grdy_action = self.greedy_action(observation)
        if training:
            #Choosing random action
            random_number = rd.random()
            if ((self.epsilon - random_number) < 0):
                return grdy_action
            else:
                random_action = rd.randint(0, self.num_actions-1)
                return random_action
        else:
            return grdy_action

    def learn(self, obs, act, rew, done, next_obs) -> None:
        """
        Update the Q-Value.

        :param obs: The observation.
        :param act: The action.
        :param rew: The reward.
        :param done: Done flag.
        :param next_obs: The next observation.
        """

        if done:
            # Decaying epsilon
            if ((self.epsilon * self.epsilon_decay) < self.epsilon_min):
                self.epsilon = self.epsilon_min
            else:
                self.epsilon = self.epsilon * self.epsilon_decay

        # Compute the loss !
        q_value = self.nn(obs)[act] #self.q_table[obs, act]
        print(q_value)
        next_max_q_value = torch.max(self.nn(next_obs)) #np.max(self.q_table[next_obs])
        print(next_max_q_value)

        #print(self.nn(obs))
        #print(self.nn(next_obs))
        #print(next_obs)
        loss1 = pow((rew + self.gamma * int(not done) * next_max_q_value - q_value), 2)
        #loss2 = np.square(rew + self.gamma * int(not done) * (np.subtract(torch.Tensor.detach(self.nn(next_obs)), torch.Tensor.detach(self.nn(next_obs))))).mean()

        print(f"loss1 = {loss1}")
        #print(f"loss2 = {loss2}")

        self.optimizer.zero_grad()
        loss1.backward()
        self.optimizer.step()

