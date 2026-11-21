"""第 24 章 · 收尾：GRPO 的组内相对优势 Â_i = (r_i − mean) / std（DeepSeekMath §4.1.2），不需要价值网络。"""

import numpy as np


def group_advantage(rewards, eps=1e-8):
    r = np.asarray(rewards, float)
    return (r - r.mean()) / (r.std() + eps)


def main():
    for rs in [[1, 0, 0, 1], [1, 0, 0, 0], [1, 1, 1, 1]]:
        print("一题 4 个回答的奖励", rs, "→ 优势", np.round(group_advantage(rs), 3))


if __name__ == "__main__":
    main()
