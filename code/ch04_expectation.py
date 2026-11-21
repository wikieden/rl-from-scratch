"""第 4 章 · 概率与期望：E[X] = Σ x · Pr(X = x)，就是按概率的加权平均；样本平均会越来越接近它。"""

import numpy as np


def expectation(values, probs):
    values, probs = np.asarray(values, float), np.asarray(probs, float)
    assert abs(probs.sum() - 1) < 1e-9, "概率要加起来等于 1"
    return float(values @ probs)


def sample_mean(values, probs, n, seed=0):
    rng = np.random.default_rng(seed)
    return float(rng.choice(values, size=n, p=probs).mean())


def main():
    vals, ps = [10, 0], [0.3, 0.7]  # 扭蛋：30% 出 10 块零食，70% 出 0
    print("期望 E =", expectation(vals, ps))
    for n in [10, 100, 10000]:
        print(f"抽 {n} 次的平均 =", round(sample_mean(vals, ps, n), 3))
    print("骰子期望 =", expectation(range(1, 7), [1 / 6] * 6))


if __name__ == "__main__":
    main()
