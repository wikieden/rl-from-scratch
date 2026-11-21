"""第 21 章 · baseline 与 reward-to-go：减掉 b 不改变梯度的期望（式 13.10），但能把方差压小。"""

import numpy as np

from ch20_policy_gradient import grad_log_softmax, softmax


def grad_samples(baseline, n=20000, seed=0, means=(10.0, 11.0)):
    """两台机器奖励都在 10 左右。返回 n 个单样本梯度估计（只看 θ[1] 那一维）。"""
    rng, theta, out = np.random.default_rng(seed), np.zeros(2), []
    pi = softmax(theta)
    for _ in range(n):
        a = rng.choice(2, p=pi)
        G = rng.normal(means[a], 1.0)
        out.append((G - baseline) * grad_log_softmax(theta, a)[1])
    return np.array(out)


def reward_to_go(rewards, gamma=1.0):
    """每一步只用它之后的奖励：G_t = Σ_{t' ≥ t} γ^{t'−t} r_{t'}。"""
    out, G = [], 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return out[::-1]


def main():
    for b in [0.0, 10.5]:
        g = grad_samples(b)
        print(f"baseline = {b}：梯度平均 {g.mean():.3f}，方差 {g.var():.2f}")
    r = [1, 0, 2, 0]
    print("奖励", r, "：整条回报都是", sum(r), "；reward-to-go", reward_to_go(r))


if __name__ == "__main__":
    main()
