from typing import Tuple, Optional
import matplotlib.pyplot as plt
import gym
import numpy as np
import copy
from gym import Env
from numpy import ndarray

from agent import DQNAgent


def run_episode(env: Env, agent: DQNAgent, training: bool, gamma) -> float:
    """
    Interact with the environment for one episode using actions derived from the q_table and the action_selector.

    :param env: The gym environment.
    :param agent: The agent.
    :param training: If true the q_table will be updated using q-learning. The flag is also passed to the action selector.
    :param gamma: The discount factor.
    :return: The cumulative discounted reward.
    """
    done = False
    obs = env.reset()
    cum_reward = 0.
    t = 0
    while not done:
        action = agent.act(obs, training)
        new_obs, reward, done, _ = env.step(action)
        if training:
            agent.learn(obs, action, reward, done, new_obs)
        #else:
            #env.render()
        obs = new_obs
        cum_reward += gamma ** t * reward
        t += 1
    return cum_reward


def train(env: Env, gamma: float, num_episodes: int, evaluate_every: int, num_evaluation_episodes: int, update_target_network_every: int,
          graph_period: int, alpha: float, epsilon_max: Optional[float] = None, epsilon_min: Optional[float] = None,
          epsilon_decay: Optional[float] = None) -> Tuple[DQNAgent, ndarray, ndarray]:
    """
    Training loop.

    :param env: The gym environment.
    :param gamma: The discount factor.
    :param num_episodes: Number of episodes to train.
    :param evaluate_every: Evaluation frequency.
    :param num_evaluation_episodes: Number of episodes for evaluation.
    :param alpha: Learning rate.
    :param epsilon_max: The maximum epsilon of epsilon-greedy.
    :param epsilon_min: The minimum epsilon of epsilon-greedy.
    :param epsilon_decay: The decay factor of epsilon-greedy.
    :return: Tuple containing the agent, the returns of all training episodes and averaged evaluation return of
            each evaluation.
    """
    digits = len(str(num_episodes))
    agent = DQNAgent(env.observation_space.shape, env.action_space.n, alpha, gamma, epsilon_max,
                          epsilon_min, epsilon_decay)
    evaluation_returns = np.zeros(num_episodes // evaluate_every)
    returns = np.zeros(num_episodes)

    for episode in range(num_episodes):
        returns[episode] = run_episode(env, agent, True, gamma)

        if (episode + 1) % evaluate_every == 0:
            evaluation_step = episode // evaluate_every
            cum_rewards_eval = np.zeros(num_evaluation_episodes)
            for eval_episode in range(num_evaluation_episodes):
                cum_rewards_eval[eval_episode] = run_episode(env, agent, False, gamma)
            evaluation_returns[evaluation_step] = np.mean(cum_rewards_eval)
            print(f"Episode {(episode + 1): >{digits}}/{num_episodes:0{digits}}:\t"
                  f"Averaged evaluation return {evaluation_returns[evaluation_step]:0.3}"
                  f"  (epsilon =  {agent.epsilon})")

        if (episode + 1) % update_target_network_every == 0:
            #agent.nn_target = agent.nn.load_state_dict(agent.nn_target) #attribute "copy" needed in agent object, don't undestand the error
            agent.nn_target = copy.deepcopy(agent.nn) #seems to work as it should

    return agent, returns, evaluation_returns


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    # test

    num_episodes = 1000
    num_samples = 3
    avg_evaluation = np.zeros((num_samples, (num_episodes // 50)))

    for x in range(num_samples):
        print(f"Training session: {x}")
        agent, returns, evaluation_returns = train(env, 0.99, num_episodes, 50, 32, 10, 100, 0.001, 1.0, 0.05, 0.99)
        #plt.plot(evaluation_returns)
        avg_evaluation[x] = evaluation_returns

    avg_evaluation_means = np.mean(avg_evaluation, axis=0)
    avg_evaluation_stds = np.std(avg_evaluation, axis=0)
    plt.plot(avg_evaluation_means, color='r')
    plt.fill_between(range(num_episodes // 50), avg_evaluation_means - avg_evaluation_stds, avg_evaluation_means + avg_evaluation_stds, alpha=.1, color='r')
    plt.ylabel(f"evaluation")
    #plt.xlabel(f"episodes x{graph_period}")
    plt.show()

