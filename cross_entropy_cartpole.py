from typing import Callable, Generator, List
import torch
from torch import nn
import gymnasium as gym
import numpy as np
from dataclasses import dataclass, field


BATCH_SIZE = 16


@dataclass
class Step:
    obs: tuple
    action: int


@dataclass
class Episode:
    steps: List[Step] = field(default_factory=list)
    total_discounted_reward: float = 0.0


def get_episode(env, get_action: Callable[[tuple], int]) -> Episode:
    episode = Episode()
    step = Step(obj=env.reset())
    while True:
        episode.steps.append(step)
        step.action = get_action(step.obs)
        new_obs, reward, terminated, truncated, _ = env.step(step.action)
        episode.total_discounted_reward += reward
        if terminated or truncated:
            break
        step = Step(obs=new_obs)
    return episode



def batch_generator(env, agent) -> Generator[List[Episode], None, None]:
    """yields batches of episodes
    infinite
    """
    sm = nn.Softmax(dim=1)

    def get_action(obs: tuple) -> int:
        input = torch.FloatTensor([obs])
        output = agent(input)
        proba = sm(output)[0]
        return np.random.choice(len(proba), p=proba)

    while True:
        yield [get_episode(env, get_action) for _ in range(BATCH_SIZE)]


def training_loop(env, agent):
    pass


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    training_loop(env)
    agent = 1