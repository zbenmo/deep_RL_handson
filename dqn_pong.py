from typing import Generator, Tuple, TypeVar
import gymnasium as gym
import torch
from torch import nn
import numpy as np
from pong_wrappers import (
    MaxAndSkipEnv,
    FireResetEnv,
    ProcessFrame84,
    ImageToPyTorch,
    BufferWrapper,
    ScaledFloatFrame,
)
from tensorboardX import SummaryWriter
from dataclasses import dataclass
from collections import deque


Action = TypeVar('Action')
State = TypeVar('State')


@dataclass
class Experience:
    state: State
    action: Action
    reward: float
    done: bool
    new_state: State


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.0
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10_000
REPLAY_START_SIZE = 10_000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01
EPSILON_DECAY_LAST_FRAME = 150_000


class Net(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


def make_env():
    env = gym.make(DEFAULT_ENV_NAME)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)


def get_experience(env, get_action) -> Generator[Tuple[Experience, float | None], None, None]:
    obs, _ = env.reset()
    total_reward = 0.0
    while True:
        action = get_action(obs)
        new_obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        done_reward = None
        if done:
            done_reward = total_reward
            total_reward = 0.0
        yield (Experience(
            state=obs,
            reward=reward,
            done = done,
            new_state = new_obs
        ), done_reward)
        obs = new_obs if not done else env.reset()[0]


device = "cpu"


def training_loop(env, agent):
    epsilon = 1.0

    def get_action(obs):
        if np.random.bernoli(epsilon):
            return env.action_space.sample()
        else:
            state_a = np.array([obs], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = agent(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            return int(act_v.item())

    behaviour = get_experience(env, get_action)

    replay_buffer = deque(
        (
            behaviour()[0] for _ in range(REPLAY_START_SIZE)
        ),
        maxlen=REPLAY_SIZE
    )

    tgt_net = Net()

    frame_idx = 0
    while True:
        frame_idx += 1
        epsilon = max(
            EPSILON_FINAL, EPSILON_START -
            frame_idx / EPSILON_DECAY_LAST_FRAME
        )

        experience, total_reward = behaviour()
        replay_buffer.append(experience)




if __name__ == "__main__":
    env = make_env()
    agent = Net()
    training_loop(env, agent)
