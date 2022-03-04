from typing import Optional
from numpy import ndarray
import numpy as np
import random as rd
import math

def create_q_table(num_states: int, num_actions: int) -> ndarray:
    """
    Function that returns a q_table as an array of shape (num_states, num_actions) filled with zeros.

    :param num_states: Number of states.
    :param num_actions: Number of actions.
    :return: q_table: Initial q_table.
    """
    # TODO: complete.
    q_table = np.zeros([num_states, num_actions])
    return q_table

class QLearnerAgent:
    """
    The agent class for exercise 1.
    """

    def __init__(self,
                 num_states: int,
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
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.q_table = create_q_table(num_states, num_actions)
        self.num_actions = num_actions
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon = epsilon_max

    def greedy_action(self, observation: int) -> int:
        """
        Return the greedy action.

        :param observation: The observation.
        :return: The action.
        """
        # TODO: complete.
        action = np.argmax(self.q_table[observation])
        return action

    def act(self, observation: int, training: bool = True) -> int:
        """
        Return the action.

        :param observation: The observation.
        :param training: Boolean flag for training, when not training agent
        should act greedily.
        :return: The action.
        """
        # TODO: complete.
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

    def learn(self, obs: int, act: int, rew: float, done: bool, next_obs: int) -> None:
        """
        Update the Q-Value.

        :param obs: The observation.
        :param act: The action.
        :param rew: The reward.
        :param done: Done flag.
        :param next_obs: The next observation.
        """
        # TODO: complete.
        if done:
            # Decaying epsilon
            if ((self.epsilon * self.epsilon_decay) < self.epsilon_min):
                self.epsilon = self.epsilon_min
            else:
                self.epsilon = self.epsilon * self.epsilon_decay

        q_value = self.q_table[obs, act] #act is a double?
        next_max_q_value = np.max(self.q_table[next_obs])

        new_state_value = q_value + self.learning_rate * (rew + self.gamma * int(not done) * next_max_q_value - q_value) #Q(s,a) <- Q(s,a) + \alpha * [r + \gamma * max_a[Q(s',a')] - Q(s,a)]
        self.q_table[obs, act] = new_state_value #act is a double?

        #print(rew) #reward always zero, why?

def _main():
    tab = create_q_table(10, 10)
    tab[2, 1] = 28
    tab[2, 2] = 29
    test = tab[2]
    print(test)
    agent_test = QLearnerAgent(10, 10, 0.01, 0.99, 1.0, 0.05, 0.99)
    print(agent_test.epsilon)

if(__name__ == "__main__"):
    _main()

