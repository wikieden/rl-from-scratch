"""第 20 章 · 策略梯度与 REINFORCE：∇J = E[G ∇log π(A|S)]（式 13.8），θ ← θ + α G ∇log π（式 13.8 的更新）。"""

import numpy as np
import torch


def softmax(theta):
    e = np.exp(theta - theta.max())
    return e / e.sum()


def grad_log_softmax(theta, a):
    """∇_θ log π(a) = one_hot(a) − π（softmax 策略的现成结论，下面用 autograd 核对）。"""
    g = -softmax(theta)
    g[a] += 1
    return g


def autograd_check(theta, a):
    t = torch.tensor(theta, requires_grad=True)
    torch.log_softmax(t, dim=0)[a].backward()
    return t.grad.numpy()


def reinforce_bandit(steps=500, alpha=0.1, seed=0, means=(1.0, 2.0)):
    """两台老虎机，策略 = softmax(θ)。一局只有一步，所以回报 G 就是这一步的奖励。"""
    rng, theta, hist = np.random.default_rng(seed), np.zeros(len(means)), []
    for _ in range(steps):
        pi = softmax(theta)
        a = rng.choice(len(means), p=pi)
        G = rng.normal(means[a], 1.0)
        theta += alpha * G * grad_log_softmax(theta, a)
        hist.append(softmax(theta)[1])
    return theta, hist


def main():
    theta = np.array([0.0, 0.0])
    print("θ = [0, 0]：π =", softmax(theta), "；∇log π(右) =", grad_log_softmax(theta, 1))
    print("autograd 核对：", autograd_check(theta, 1))
    th, hist = reinforce_bandit()
    print("REINFORCE 500 步后 π(好机器) =", round(hist[-1], 3), "；前几步", [round(h, 3) for h in hist[:3]])


if __name__ == "__main__":
    main()
