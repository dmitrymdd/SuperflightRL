"""
Диагностика: одинаковы ли два прогона при одинаковых действиях, и откуда берётся расхождение.

Для каждой карты:
  A) прогон 1 -> тот же сид сразу ещё раз (карта НЕ перегенерируется, тот же уровень в памяти)
  B) тот же сид после захода на другую карту (карта ПЕРЕГЕНЕРИРУЕТСЯ из сида)
Сравниваем наблюдения (скорость, скорость в системе игрока, гравитация, угловая скорость) шаг за шагом
и печатаем, на каком решении они впервые разошлись.

Как читать:
  - A расходится, B тоже          -> недетерминированность в самой физике/тайминге игры;
  - A совпадает, B расходится     -> карта по одному сиду получается разной (сид не задаёт карту однозначно);
  - всё совпадает                 -> всё детерминировано.
Запуск: игра открыта -> python bench_determinism.py (без аргументов).
"""
import traceback

import numpy as np

from superflight_env import SuperflightEnv

PORT = 5555
SEEDS = [11, 22]
OTHER_SEED = 99
SEQ_LEN = 500
TOL = 2e-3          # наблюдения печатаются с 4 знаками после запятой


def run(env, seed, actions):
    env.reset(options={"map_seed": seed})
    traj = []
    for a in actions:
        obs, r, done, trunc, info = env.step(a)
        traj.append(obs[:10].copy())
        if done or trunc:
            break
    return np.array(traj)


def compare(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return "пусто"
    diff = np.abs(a[:n] - b[:n]).max(axis=1)
    bad = np.where(diff > TOL)[0]
    if len(bad) == 0 and len(a) == len(b):
        return "совпали полностью (%d решений)" % n
    if len(bad) == 0:
        return "совпали на общей части, но длины разные: %d и %d" % (len(a), len(b))
    i = int(bad[0])
    return "первое расхождение на решении %d из %d/%d (разница %.3f)" % (i, len(a), len(b), diff[i])


def main():
    env = SuperflightEnv(port=PORT, frame_skip=4, max_decisions=SEQ_LEN)
    env._cmd("CONFIG render 0")
    env._cmd("CONFIG capture 50")
    rng = np.random.default_rng(7)
    for s in SEEDS:
        acts = [rng.integers(0, 3, size=2) for _ in range(SEQ_LEN)]
        print("\n=== карта %d ===" % s)
        env.reset(options={"map_seed": OTHER_SEED})            # чтобы первый прогон точно генерировал карту
        t1 = run(env, s, acts)
        t2 = run(env, s, acts)                                 # тот же сид подряд: карта не меняется
        print("A (тот же уровень в памяти):      ", compare(t1, t2))
        env.reset(options={"map_seed": OTHER_SEED})
        t3 = run(env, s, acts)                                 # карта перегенерирована из сида
        print("B (карта сгенерирована заново):   ", compare(t1, t3))
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n!!! Ошибка:")
        traceback.print_exc()
    input("\nНажми Enter, чтобы закрыть окно...")
