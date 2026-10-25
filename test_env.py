"""
Проверка связки Python <-> игра. Аргументы не обязательны:
просто запусти скрипт, и он сам прогонит два режима (с графикой и без) и сравнит скорость.

1) Запусти игру (плагин поднимает сервер сам), можно оставаться в меню.
2) python test_env.py
Необязательно: --render 0 | 1 | both  (по умолчанию both), --episodes N, --policy random|straight
"""
import argparse
import time

import numpy as np

from superflight_env import SuperflightEnv

# Если запускаешь без аргументов (кнопка Run), меняй настройки здесь:
CAPTURE_HZ = 50


def run_block(env, label, episodes, policy, frame_skip):
    print("\n=== %s ===" % label)
    total_dec, total_game, t_play = 0, 0.0, 0.0
    for ep in range(episodes):
        t0 = time.time()
        obs, info = env.reset()
        t_reset = time.time() - t0
        if np.allclose(obs, 0.0):
            print("ВНИМАНИЕ: первый obs после reset состоит из нулей (ep %d)" % ep)
        if ep == 0 and label.startswith("render=1"):
            print("Первый obs (первые 15):", np.round(obs[:15], 3))
            print("Лучи: min=%.2f mean=%.2f (1.0 = свободно)" % (obs[15:].min(), obs[15:].mean()))
        done = truncated = False
        n = 0
        t1 = time.time()
        while not (done or truncated):
            if policy == "random":
                action = env.action_space.sample()
            else:
                action = np.array([1, 1])  # ничего не нажимать
            obs, r, done, truncated, info = env.step(action)
            n += 1
        dt = time.time() - t1
        game_s = n * frame_skip * env.fixed_dt
        total_dec += n
        total_game += game_s
        t_play += dt
        print("ep %d | reset %.2fs | %4d решений | счёт %7.0f | игровых сек %5.1f | %4.0f реш/с | %.1fx реального"
              % (ep, t_reset, n, info["score"], game_s, n / max(dt, 1e-9), game_s / max(dt, 1e-9)))
    speed = total_game / max(t_play, 1e-9)
    print("-> %s: %.0f реш/с, %.1fx реального времени" % (label, total_dec / max(t_play, 1e-9), speed))
    return speed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="*", default=[11, 22, 33])
    ap.add_argument("--policy", choices=["random", "straight"], default="random")
    ap.add_argument("--render", choices=["0", "1", "both"], default="both")
    # частота кадров игры в режиме «1 кадр = фиксированный шаг времени».
    # 50 = точно как физика (самый честный счёт), 25 = в 2 раза меньше кадров, 0 = выключить режим
    ap.add_argument("--capture", type=int, default=CAPTURE_HZ)
    args = ap.parse_args()

    env = SuperflightEnv(port=args.port, frame_skip=args.frame_skip, map_seeds=args.seeds)
    print("Подключились. obs_dim=%d, fixed_dt=%.3f, frame_skip=%d" %
          (env.obs_dim, env.fixed_dt, args.frame_skip))

    env._cmd("CONFIG capture %d" % args.capture)
    print("captureFramerate = %d" % args.capture)

    modes = ["1", "0"] if args.render == "both" else [args.render]
    results = {}
    for m in modes:
        env._cmd("CONFIG render %s" % m)
        results[m] = run_block(env, "render=%s" % m, args.episodes, args.policy, args.frame_skip)

    if len(results) == 2:
        print("\nСравнение: с графикой %.1fx, без графики %.1fx (выигрыш %.2f раза)"
              % (results["1"], results["0"], results["0"] / max(results["1"], 1e-9)))
    env.close()


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        print("\n!!! Ошибка:")
        traceback.print_exc()
    input("\nНажми Enter, чтобы закрыть окно...")
