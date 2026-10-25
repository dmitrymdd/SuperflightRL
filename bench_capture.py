"""
Проверка: можно ли ускорить игру, снижая частоту кадров (captureFramerate), не испортив счёт.

Идея: физика всегда идёт с шагом 0.02 с, а очки считаются раз за кадр. Если на одной и той же карте
подать ОДИНАКОВУЮ последовательность действий, траектория должна получиться одинаковой при любой
частоте кадров. Значит, различия в счёте и времени жизни = искажение от частоты кадров.
Первая строка (50 Гц) повторяется дважды: так видно, насколько результаты отличаются сами от себя.

Запуск: игра открыта -> python bench_capture.py (без аргументов).
"""
import time
import traceback

import numpy as np

from superflight_env import SuperflightEnv

HZ_LIST = [50, 50, 25, 10]      # 50 повторяется специально
SEEDS = [11, 22, 33, 44]
SEQ_LEN = 700                   # решений на эпизод (потолок)
FRAME_SKIP = 4
PORT = 5555


def run_one(env, hz, seed, actions):
    obs, info = env.reset(options={"map_seed": seed})
    t0 = time.time()
    n = 0
    done = False
    info = {}
    for a in actions:
        obs, r, done, trunc, info = env.step(a)
        n += 1
        if done or trunc:
            break
    dt = time.time() - t0
    ep_sec = info.get("physics_steps", n * FRAME_SKIP) * env.fixed_dt
    return {
        "hz": hz, "seed": seed, "n": n, "died": bool(done),
        "score": float(info.get("score", 0.0)), "ep_sec": ep_sec,
        "speed": (n * FRAME_SKIP * env.fixed_dt) / max(dt, 1e-9),
    }


def main():
    env = SuperflightEnv(port=PORT, frame_skip=FRAME_SKIP, max_decisions=SEQ_LEN)
    env._cmd("CONFIG render 0")
    rng = np.random.default_rng(123)
    seqs = {s: [rng.integers(0, 3, size=2) for _ in range(SEQ_LEN)] for s in SEEDS}

    rows = []
    for i, hz in enumerate(HZ_LIST):
        env._cmd("CONFIG capture %d" % hz)
        print("\n--- capture %d Гц (прогон %d) ---" % (hz, i + 1))
        for s in SEEDS:
            r = run_one(env, hz, s, seqs[s])
            rows.append(r)
            print("seed %2d | %3d реш | %s | счёт %6.0f | жизнь %5.1f с | %.1fx реального"
                  % (s, r["n"], "умер " if r["died"] else "живой", r["score"], r["ep_sec"], r["speed"]))

    print("\n=== Итог (относительно первого прогона 50 Гц) ===")
    base = {r["seed"]: r for r in rows[:len(SEEDS)]}
    idx = 0
    for i, hz in enumerate(HZ_LIST):
        block = rows[i * len(SEEDS):(i + 1) * len(SEEDS)]
        ratios, same_len = [], 0
        for r in block:
            b = base[r["seed"]]
            if b["score"] > 0:
                ratios.append(r["score"] / b["score"])
            if abs(r["n"] - b["n"]) <= 1:
                same_len += 1
        print("%2d Гц (прогон %d): скорость %.1fx | счёт/базовый: %s | одинаковая длина жизни: %d из %d"
              % (hz, i + 1, np.mean([r["speed"] for r in block]),
                 " ".join("%.2f" % x for x in ratios) if ratios else "нет данных",
                 same_len, len(block)))
    print("\nКак читать: у прогонов 50 Гц отношение должно быть ~1.00 и длина жизни одинаковая.")
    print("Если у 25 и 10 Гц отношение заметно отличается от 1 (и от разброса между двумя 50 Гц),")
    print("значит, ускорение портит награду, и пользоваться им нужно осторожно.")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n!!! Ошибка:")
        traceback.print_exc()
    input("\nНажми Enter, чтобы закрыть окно...")
