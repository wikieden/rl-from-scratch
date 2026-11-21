"""第 22 章 · Actor-Critic、优势函数与 GAE：δ_t = r_t + γV(s_{t+1}) − V(s_t)；Â_t = Σ (γλ)^l δ_{t+l}（GAE 论文式 16）。"""

import numpy as np


def td_errors(rewards, values, gamma):
    """values 比 rewards 多一个（最后一个是结束后的状态价值，结束时填 0）。"""
    r, v = np.asarray(rewards, float), np.asarray(values, float)
    return r + gamma * v[1:] - v[:-1]


def gae(rewards, values, gamma, lam):
    delta, A, out = td_errors(rewards, values, gamma), 0.0, []
    for d in reversed(delta):
        A = d + gamma * lam * A
        out.append(A)
    return np.array(out[::-1])


def mc_advantage(rewards, values, gamma):
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return np.array(out[::-1]) - np.asarray(values[:-1], float)


def main():
    r, v, g = [0, 0, 1], [0.5, 0.6, 0.8, 0.0], 0.9
    print("δ =", np.round(td_errors(r, v, g), 3))
    for lam in [0.0, 0.95, 1.0]:
        print(f"λ = {lam}：Â =", np.round(gae(r, v, g, lam), 3))
    print("MC 优势 G − V =", np.round(mc_advantage(r, v, g), 3))


if __name__ == "__main__":
    main()
