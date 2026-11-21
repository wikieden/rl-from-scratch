"""第 19 章 · 过估计与 Double DQN：max 会挑到"运气好"的估计（§6.7）；一个网络挑动作、另一个网络打分。"""

import numpy as np


def max_bias(n_actions=10, n_samples=5, trials=20000, seed=0):
    """每个动作真实价值都是 0，估计 = n_samples 次带噪声（标准差 1）的平均。"""
    rng = np.random.default_rng(seed)
    single, double = [], []
    for _ in range(trials):
        Q1 = rng.normal(0, 1, (n_actions, n_samples)).mean(axis=1)
        Q2 = rng.normal(0, 1, (n_actions, n_samples)).mean(axis=1)
        single.append(Q1.max())  # 同一组估计既挑又打分
        double.append(Q2[np.argmax(Q1)])  # Q1 挑动作，Q2 打分
    return float(np.mean(single)), float(np.mean(double))


def targets(r, gamma, q_online_next, q_target_next):
    """DQN 目标用目标网络自己挑 + 自己打分；Double DQN 用在线网络挑、目标网络打分。"""
    dqn = r + gamma * q_target_next.max()
    a = int(np.argmax(q_online_next))
    ddqn = r + gamma * q_target_next[a]
    return dqn, ddqn


def main():
    s, d = max_bias()
    print(f"10 个动作真值都是 0：单估计 E[max] = {s:.3f}，双估计 = {d:.3f}")
    dqn, ddqn = targets(0.0, 0.9, np.array([1.0, 2.0, 0.5]), np.array([1.2, 0.7, 1.5]))
    print(f"手算一步：DQN 目标 = {dqn:.2f}，Double DQN 目标 = {ddqn:.2f}")


if __name__ == "__main__":
    main()
