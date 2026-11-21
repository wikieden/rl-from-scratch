"""第 3 章 · 折扣：G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...（式 3.8），倒着算 G_t = R_{t+1} + γG_{t+1}（式 3.9）。"""


def discounted_return(rewards, gamma):
    """按定义一项一项加：第 k 个奖励乘 γ^k。"""
    return sum(gamma**k * r for k, r in enumerate(rewards))


def returns_backward(rewards, gamma):
    """从最后往前推：G_T = 0，G_t = R_{t+1} + γ G_{t+1}。返回 [G_0, ..., G_T]。"""
    G = [0.0] * (len(rewards) + 1)
    for t in reversed(range(len(rewards))):
        G[t] = rewards[t] + gamma * G[t + 1]
    return G


def main():
    r = [1, 0, 1]
    print("γ=0.9，奖励 1, 0, 1：G0 =", round(discounted_return(r, 0.9), 4))
    print("倒着算 G0..G3 =", [round(g, 4) for g in returns_backward(r, 0.9)])
    print("γ=0（近视）G0 =", discounted_return(r, 0.0), "；γ=1（不打折）G0 =", discounted_return(r, 1.0))
    long_sum = discounted_return([1] * 1000, 0.9)
    print("每步都 +1、γ=0.9，加 1000 项 =", round(long_sum, 6), "；公式 1/(1−γ) =", round(1 / (1 - 0.9), 6))


if __name__ == "__main__":
    main()
