"""第 14 章 · 蒙特卡洛 MC：一局走完，用真实回报的平均估计价值（§5.1）；常数步长版 V ← V + α(G − V)（式 6.1）。"""

import numpy as np

from rl_envs import WALK_NAMES, WALK_TRUE, random_walk_episode


def mc_episode_update(V, episode, alpha=None, counts=None, gamma=1.0):
    """first-visit：每个状态只用它第一次出现之后的回报。alpha=None 时用真平均（1/N）。"""
    G, returns = 0.0, {}
    for s, r, _ in reversed(episode):
        G = r + gamma * G
        returns[s] = G  # 倒着走，最后留下的就是第一次出现时的回报
    for s, G in returns.items():
        if alpha is None:
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
        else:
            V[s] += alpha * (G - V[s])
    return V


def mc_prediction(n_episodes, alpha=None, seed=0, init=0.5):
    rng = np.random.default_rng(seed)
    V, N = np.full(5, init), np.zeros(5)
    for _ in range(n_episodes):
        mc_episode_update(V, random_walk_episode(rng), alpha, N)
    return V


def rms(V):
    return float(np.sqrt(np.mean((V - WALK_TRUE) ** 2)))


def main():
    ep = random_walk_episode(np.random.default_rng(3))
    print("一局：", " → ".join(WALK_NAMES[s] for s, _, _ in ep), "，最后奖励", ep[-1][1])
    V = mc_prediction(1000)
    print("MC 1000 局（真平均）：", np.round(V, 3), "；真值", np.round(WALK_TRUE, 3))
    print("RMS 误差 =", round(rms(V), 4))


if __name__ == "__main__":
    main()
