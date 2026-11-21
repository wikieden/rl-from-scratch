"""第 23 章 · 从 TRPO 到 PPO：比率 r = π_新/π_旧，L = min(r·A, clip(r, 1−ε, 1+ε)·A)（PPO 论文式 7）。"""

import torch


def ppo_term(ratio, adv, eps=0.2):
    r = torch.as_tensor(ratio, dtype=torch.float32)
    return torch.minimum(r * adv, torch.clamp(r, 1 - eps, 1 + eps) * adv)


def grad_wrt_ratio(ratio, adv, eps=0.2):
    r = torch.tensor(float(ratio), requires_grad=True)
    ppo_term(r, adv, eps).backward()
    return r.grad.item()


def importance_weighted_mean(rewards, p_new, p_old):
    """重要性采样：用旧策略的样本估新策略的平均，每个样本乘 p_新 / p_旧。"""
    return sum(r * pn / po for r, pn, po in zip(rewards, p_new, p_old)) / len(rewards)


def main():
    print("重要性采样例子 =", importance_weighted_mean([1, 0], [0.8, 0.2], [0.5, 0.5]))
    for ratio, adv in [(1.1, 2.0), (1.5, 2.0), (0.9, -1.0), (0.5, -1.0), (0.5, 2.0)]:
        print(f"r = {ratio}，A = {adv:+}：L = {ppo_term(ratio, adv).item():+.2f}，dL/dr = {grad_wrt_ratio(ratio, adv):+.1f}")


if __name__ == "__main__":
    main()
