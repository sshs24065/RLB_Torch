import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch import nn, optim
from random import sample
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import QValueActor
from torchrl.objectives import DQNLoss
from torchrl.data import DiscreteTensorSpec
from torchrl.envs.libs.gym import GymEnv
from matplotlib import pyplot as plt

env = GymEnv('CartPole-v1')

model = TensorDictSequential(
    TensorDictModule(nn.Linear(4, 10), in_keys=['observation'], out_keys=['in1']),
    TensorDictModule(nn.ReLU(), in_keys=['in1'], out_keys=['in2']),
    TensorDictModule(nn.Linear(10, 6), in_keys=['in2'], out_keys=['in3']),
    TensorDictModule(nn.ReLU(), in_keys=['in3'], out_keys=['in4']),
    TensorDictModule(nn.Linear(6, 2), in_keys=['in4'], out_keys=['action_value']))
actor = QValueActor(module=model, spec=DiscreteTensorSpec(2))
loss_fn = DQNLoss(actor, action_space='categorical')
optimizer = optim.Adam(model.parameters(), lr=1e-3)
replay_buffer = []

EPOCHS = 100000
MAX_STEPS = 500
FAIL_REWARD = torch.tensor(-1000.0)
GAMMA = 0.95

train_steps = []

for i in range(EPOCHS):
    model.train()
    state = env.reset()
    t = MAX_STEPS
    with torch.no_grad():
        state = state['observation']
        for j in range(MAX_STEPS):
            action = actor(TensorDict({'observation': state}))['action']
            step_res = env.step(TensorDict({'action': action}, []))
            next_state, reward, finish = step_res['next']['observation'], step_res['next']['reward'], step_res['next']['done']
            if finish.item():
                reward = FAIL_REWARD
                replay_buffer.append([state, action, next_state, finish, reward])
                t = j
                break
            replay_buffer.append([state, action, next_state, finish, reward])
            state = next_state
        reward_backward = torch.tensor(0.0)
        for j in range(t - 1, -1, -1):
            reward_backward, replay_buffer[j][4] = (replay_buffer[j][4] + reward_backward) * GAMMA, replay_buffer[j][4] + reward_backward
    model.zero_grad()
    for j in sample(replay_buffer, len(replay_buffer) // 2):
        loss = loss_fn(
            observation=j[0],
            action=j[1],
            next_observation=j[2],
            next_done=torch.tensor([1]) if j[3].item() else torch.tensor([0]),
            next_reward=torch.tensor([j[4]]))
        loss.backward()
        optimizer.step()
    model.eval()
    env.reset()
    state = env.reset()
    t = MAX_STEPS
    with torch.no_grad():
        state = state['observation']
        for j in range(MAX_STEPS):
            action = actor(TensorDict({'observation': state}))['action']
            step_res = env.step(TensorDict({'action': action}, []))
            next_state, reward, finish = step_res['next']['observation'], step_res['next']['reward'], step_res['next']['done']
            if finish.item():
                reward = FAIL_REWARD
                t = j
                break
            state = next_state
    train_steps.append(t)
    print(i)
    replay_buffer.clear()
plt.plot([i for i in range(EPOCHS)], train_steps)
plt.show()
