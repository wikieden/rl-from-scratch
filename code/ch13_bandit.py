"""第 13 章 · 探索与利用：多臂老虎机、ε-greedy、增量平均 Q_{n+1} = Q_n + (R_n − Q_n)/n（式 2.3）。"""

import numpy as np

TRUE_MEANS = np.array([1.0, 1.5, 0.5])  # 三台机器的真实平均奖励（讲义编的），第 2 台最好


def incremental_mean(rewards):
    q, hist = 0.0, []
    for n, r in enumerate(rewards, start=1):
        q = q + (r - q) / n
        hist.append(q)
    return hist


def run_bandit(eps, steps=1000, seed=0, means=TRUE_MEANS):
    rng = np.random.default_rng(seed)
    Q, N = np.zeros(len(means)), np.zeros(len(means))
    total, best = 0.0, 0
    for _ in range(steps):
        a = rng.integers(len(means)) if rng.random() < eps else int(np.argmax(Q))
        r = rng.normal(means[a], 1.0)
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
        total += r
        best += a == np.argmax(means)
    return total / steps, best / steps, Q


def main():
    print("增量平均 [2, 4, 6, 8]：", incremental_mean([2, 4, 6, 8]))
    for eps in [0.0, 0.1]:
        res = [run_bandit(eps, seed=s) for s in range(200)]
        avg_r = np.mean([r[0] for r in res])
        pct = np.mean([r[1] for r in res])
        print(f"ε = {eps}：200 次实验平均每步奖励 {avg_r:.3f}，选到最好机器的比例 {pct:.0%}")


if __name__ == "__main__":
    main()
