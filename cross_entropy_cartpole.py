from typing import Callable, Final, Generator, List, Tuple
import torch
from torch import nn
from torch import optim
import gymnasium as gym
import numpy as np
from dataclasses import dataclass, field
from tensorboardX import SummaryWriter
from itertools import chain


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


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
    obs, _ = env.reset()
    while True:
        step = Step(obs=obs, action=get_action(obs))
        episode.steps.append(step)
        new_obs, reward, terminated, truncated, _ = env.step(step.action)
        episode.total_discounted_reward += reward
        if terminated or truncated:
            break
        obs=new_obs
    return episode



def batch_generator(env, agent) -> Generator[List[Episode], None, None]:
    """yields batches of episodes
    infinite
    """
    sm = nn.Softmax(dim=1)

    def get_action(obs: tuple) -> int:
        input = torch.FloatTensor([obs])
        output = agent(input)
        proba = sm(output).data.numpy()[0]
        return np.random.choice(len(proba), p=proba)

    while True:
        yield [get_episode(env, get_action) for _ in range(BATCH_SIZE)]


def filter_batch(batch: List[Episode]) -> Tuple[List[Episode], float, float]:
    """
    Returns:
        elite_episodes
        reward_abound
        reward_mean
    """
    rewards = [episode.total_discounted_reward for episode in batch]
    reward_bound = np.percentile(rewards, PERCENTILE)
    reward_mean = np.mean(rewards)
    elite_episodes = [
        episode for episode in batch if episode.total_discounted_reward >= reward_bound
    ]
    return elite_episodes, reward_bound, reward_mean


def training_loop(env, agent):
    """"""
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=agent.parameters(), lr=0.01)

    with SummaryWriter(comment="-cartpole") as writer:
        for iter_no, batch in enumerate(batch_generator(env, agent)):
            elite_episodes, reward_b, reward_m = filter_batch(batch)
            all_steps = (episode.steps for episode in elite_episodes)
            obs_v, acts_v = list(zip(
                *((step.obs, step.action) for step in chain.from_iterable(all_steps))
            ))

            optimizer.zero_grad()
            action_scores_v = agent(torch.FloatTensor(obs_v))
            loss_v = objective(action_scores_v, torch.LongTensor(acts_v))
            loss_v.backward()
            optimizer.step()

            print((
                "%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" %
                (iter_no, loss_v.item(), reward_m, reward_b)
            ))
            writer.add_scalar("loss", loss_v.item(), iter_no)
            writer.add_scalar("reward_bound", reward_b, iter_no)
            writer.add_scalar("reward_mean", reward_m, iter_no)
            if reward_m > 199:
                print("Solved!")
                break


if __name__ == "__main__":
    env = gym.make('CartPole-v1') # v0?
    obs_size: Final[int] = env.observation_space.shape[0]
    n_actions: Final[int] = env.action_space.n
    agent = Net(obs_size, HIDDEN_SIZE, n_actions)
    training_loop(env, agent)
