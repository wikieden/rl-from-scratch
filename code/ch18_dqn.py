"""第 18 章 · DQN：神经网络 Q(s, a; θ) + 经验回放 + 目标网络；loss = (y − Q(s, a))²，y = r + γ max_a' Q_目标(s', a')。"""

import random

import torch
from torch import nn

from rl_envs import CORRIDOR_N, corridor_step

GAMMA = 0.9


def one_hot(s):
    x = torch.zeros(CORRIDOR_N)
    x[s] = 1.0
    return x


def td_target(r, q_next_target, done, gamma=GAMMA):
    return r + (0.0 if done else gamma * q_next_target.max().item())


def train_dqn(steps=3000, seed=0, eps=0.3, batch=32, sync_every=100, lr=0.01):
    torch.manual_seed(seed)
    random.seed(seed)
    q = nn.Sequential(nn.Linear(CORRIDOR_N, 32), nn.ReLU(), nn.Linear(32, 2))
    target = nn.Sequential(nn.Linear(CORRIDOR_N, 32), nn.ReLU(), nn.Linear(32, 2))
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buffer, s = [], 0
    for t in range(steps):
        with torch.no_grad():
            a = random.randrange(2) if random.random() < eps else int(q(one_hot(s)).argmax())
        ns, r, done = corridor_step(s, a)
        buffer.append((s, a, r, ns, done))
        buffer = buffer[-2000:]  # 经验回放：存最近 2000 条
        s = 0 if done else ns
        if len(buffer) >= batch:
            b = random.sample(buffer, batch)  # 随机抽，打乱前后相关
            S = torch.stack([one_hot(x[0]) for x in b])
            A = torch.tensor([x[1] for x in b])
            R = torch.tensor([x[2] for x in b])
            NS = torch.stack([one_hot(x[3]) for x in b])
            D = torch.tensor([float(x[4]) for x in b])
            with torch.no_grad():
                y = R + GAMMA * (1 - D) * target(NS).max(dim=1).values  # 目标网络算 y，不求梯度
            pred = q(S).gather(1, A[:, None]).squeeze(1)
            loss = ((y - pred) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if t % sync_every == 0:
            target.load_state_dict(q.state_dict())  # 隔一阵子才把目标网络对齐
    return q


def main():
    print("手算 y：r = 0，Q_目标(s') = [0.5, 0.8]，y =", td_target(0.0, torch.tensor([0.5, 0.8]), False))
    q = train_dqn()
    with torch.no_grad():
        Q = torch.stack([q(one_hot(s)) for s in range(CORRIDOR_N - 1)])
    print("学到的 Q（每行 [左, 右]）：\n", Q.numpy().round(3))
    print("理论 Q(s, 右) = γ^(4−s)：", [round(GAMMA ** (4 - s), 3) for s in range(CORRIDOR_N - 1)])


if __name__ == "__main__":
    main()
