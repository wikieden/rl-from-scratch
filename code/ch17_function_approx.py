"""第 17 章 · 函数近似与梯度下降：v̂(s, w) 代替表格；w ← w + α[U − v̂(S, w)]∇v̂(S, w)（式 9.7）。"""

import numpy as np
import torch

from rl_envs import WALK_TRUE, random_walk_episode


def slope(f, x, h=1e-4):
    """数值求导：往右挪一点点，看 f 变了多少。"""
    return (f(x + h) - f(x - h)) / (2 * h)


def gradient_descent(w0=0.0, lr=0.1, steps=30):
    """最小化 (w − 3)²：导数 2(w − 3)，每步 w ← w − lr · 导数。"""
    w, hist = w0, [w0]
    for _ in range(steps):
        w -= lr * 2 * (w - 3)
        hist.append(w)
    return hist


def features(s):
    """随机游走 5 个状态，用 2 个特征 [1, 位置] 表示（位置 A=−1 … E=+1）。"""
    return np.array([1.0, (s - 2) / 2])


def gradient_mc_linear(n_episodes=5000, alpha=0.02, seed=0):
    """线性 v̂(s, w) = w · x(s)，梯度就是 x(s)。目标 U 用整局回报 G（梯度 MC，§9.3）。
    步长随局数慢慢变小（α_k = α / (1 + k/100)），否则 w 会一直在真值附近抖。"""
    rng, w = np.random.default_rng(seed), np.zeros(2)
    for k in range(n_episodes):
        a = alpha / (1 + k / 100)
        ep = random_walk_episode(rng)
        G = ep[-1][1]  # γ = 1，中间奖励都是 0，回报就是最后那个奖励
        for s, _, _ in ep:
            x = features(s)
            w += a * (G - w @ x) * x
    return w


def main():
    print("f(x) = x² 在 x = 3 的斜率 ≈", round(slope(lambda x: x * x, 3.0), 4))
    hist = gradient_descent()
    print("梯度下降 (w − 3)²：", [round(h, 3) for h in hist[:5]], "…", round(hist[-1], 4))
    x = torch.tensor(3.0, requires_grad=True)
    (x**2).backward()
    print("PyTorch autograd：d(x²)/dx 在 3 =", x.grad.item())
    w = gradient_mc_linear()
    print("线性近似 w =", np.round(w, 3), "；v̂ =", np.round([w @ features(s) for s in range(5)], 3))
    print("真值       ", np.round(WALK_TRUE, 3))


if __name__ == "__main__":
    main()
