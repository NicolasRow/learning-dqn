import random
from typing import Optional
from numpy import ndarray
import numpy as np
from model import QModel
from replay_buffer import ReplayBuffer
import torch
import random as rd

class DQNAgent:
    """
    The agent class for exercise 1.
    """

    def __init__(self,
                 obs_shape: tuple,
                 num_actions: int,
                 learning_rate: float,
                 gamma: float,
                 epsilon_max: Optional[float] = None,
                 epsilon_min: Optional[float] = None,
                 epsilon_decay: Optional[float] = None,
                 capacity: Optional[int] = 10000):
        """
        :param num_states: Number of states.
        :param num_actions: Number of actions.
        :param learning_rate: The learning rate.
        :param gamma: The discount factor.
        :param epsilon_max: The maximum epsilon of epsilon-greedy.
        :param epsilon_min: The minimum epsilon of epsilon-greedy.
        :param epsilon_decay: The decay factor of epsilon-greedy.
        """
        self.obs_shape = obs_shape
        self.capacity = capacity
        self.obs_dim = int(np.prod(self.obs_shape))
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.nn = QModel(self.obs_dim, num_actions)
        self.nn_target = QModel(self.obs_dim, num_actions)
        self.rb = ReplayBuffer(capacity, self.obs_shape)
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
        action = torch.argmax(self.nn(observation)).item()
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

        self.rb.add_transition(obs, act, rew, done, next_obs)

        if done:
            # Decaying epsilon
            if ((self.epsilon * self.epsilon_decay) < self.epsilon_min):
                self.epsilon = self.epsilon_min
            else:
                self.epsilon = self.epsilon * self.epsilon_decay

        states, actions, rewards, dones, next_states = self.rb.sample(64)
        # if len(self.rb) < 16:
        #     observations = self.rb.sample(16)
        # else:
        #     observations = self.rb.sample(16)

        # Compute the loss !
        states_nn = self.nn(states) #Q_Values without action
        next_pred_states_nn = self.nn(next_states)

        q_values = torch.from_numpy(np.zeros(len(actions))) #Selected Q_Values by actions

        x = 0
        while x < len(actions):
            q_values[x] = states_nn[x][actions[x]]
            x += 1

        new_actions = torch.argmax(next_pred_states_nn, 1)

        with torch.no_grad():
            next_target_states_nn = self.nn_target(next_states)

        target_q_values = torch.from_numpy(np.zeros(len(actions)))

        y = 0
        while y < len(actions):
            target_q_values[y] = next_target_states_nn[y][new_actions[y]]
            y += 1

        loss = pow(((torch.from_numpy(rewards) + self.gamma * torch.from_numpy(1-dones) * target_q_values) - q_values), 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

