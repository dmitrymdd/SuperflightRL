"""
Первое обучение PPO (Stable-Baselines3) на Superflight.

Перед запуском:  pip install stable-baselines3 gymnasium numpy
Игра должна быть запущена (плагин Superflight AI поднимает сервер). Аргументы не нужны:
настройки в блоке ниже. Скрипт:
  1) считает базовую случайную стратегию на контрольных картах,
  2) обучает PPO на ДРУГИХ картах (ускоренный режим кадров),
  3) проверяет обученную модель на контрольных картах в ЧЕСТНОМ режиме (50 Гц, как в обычной игре).
Контрольные карты (EVAL_SEEDS) агент во время обучения не видит.
"""
import os
import time
import traceback

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from superflight_env import SuperflightEnv

# ---------------- настройки ----------------
PORT = 5555
TOTAL_DECISIONS = 20_000         # при RESUME_FROM это ДОБАВКА к num_timesteps модели (80k + 20k = 100k)
FRAME_SKIP = 4                   # физических шагов на решение
CAPTURE_HZ = 25                  # частота кадров во время ОБУЧЕНИЯ: 25 ~1.7x быстрее, чем 50 (счёт слегка занижается)
EVAL_CAPTURE_HZ = 50             # частота кадров при ПРОВЕРКЕ: 50 = честный счёт игры
RENDER = 0                       # 0 = без отрисовки
GAMMA = 0.995
TRAIN_SEEDS = list(range(100, 140))   # 40 карт для обучения
EVAL_SEEDS = [1, 2, 3, 4, 5]          # контрольные карты
EVAL_REPEATS = 2                      # сколько раз пройти каждую
RESUME_FROM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "20261015-142950", "ppo_80000_steps.zip")   # None = обучение с нуля
VECNORM_FROM = None              # путь к .pkl со статистикой VecNormalize (ppo_vecnormalize_<N>_steps.pkl); None = новая
# -------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_ROOT = os.path.join(HERE, "runs")


class ScoreLogger(BaseCallback):
    """Печатает средний счёт игры за последние эпизоды (а не только масштабированную награду)."""

    def __init__(self, every=10):
        super().__init__()
        self.every = every
        self.scores, self.lens = [], []
        self.t0 = time.time()

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if done:
                self.scores.append(float(info.get("score", 0.0)))
                self.lens.append(int(info.get("physics_steps", 0)))
                if len(self.scores) % self.every == 0:
                    last = self.scores[-self.every:]
                    ll = self.lens[-self.every:]
                    print("[эпизодов %4d | решений %6d | %4.0f мин] счёт: среднее %.0f, медиана %.0f, лучший %.0f | жизнь ~%.1f с"
                          % (len(self.scores), self.num_timesteps, (time.time() - self.t0) / 60.0,
                             np.mean(last), np.median(last), np.max(last), np.mean(ll) * 0.02))
        return True


def evaluate(env, policy_fn, seeds, repeats, label):
    scores, lives = [], []
    for _ in range(repeats):
        for s in seeds:
            obs, info = env.reset(options={"map_seed": s})
            done = trunc = False
            info = {}
            while not (done or trunc):
                obs, r, done, trunc, info = env.step(policy_fn(obs))
            scores.append(float(info.get("score", 0.0)))
            lives.append(info.get("physics_steps", 0) * env.fixed_dt)
    print("%s: средний счёт %.0f (медиана %.0f, ±%.0f), лучший %.0f, средняя жизнь %.1f с"
          % (label, np.mean(scores), np.median(scores), np.std(scores), np.max(scores), np.mean(lives)))
    return float(np.mean(scores))


def make_run_dir():
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(RUN_ROOT, stamp)
    try:
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
    except OSError as e:
        print("Не удалось создать папку %s: %r" % (run_dir, e))
        fallback = os.path.join(os.path.expanduser("~"), "superflight_runs", stamp)
        os.makedirs(fallback, exist_ok=True)
        print("Использую запасную папку:", fallback)
        return fallback


def main():
    env = SuperflightEnv(port=PORT, frame_skip=FRAME_SKIP, map_seeds=TRAIN_SEEDS,
                         max_decisions=3000, extra_config={"render": RENDER})
    print("Подключились. obs_dim=%d" % env.obs_dim)
    run_dir = make_run_dir()
    print("Результаты пишутся в:", run_dir)

    print("\n--- Базовая стратегия: случайные действия (режим %d Гц) ---" % EVAL_CAPTURE_HZ)
    env._cmd("CONFIG capture %d" % EVAL_CAPTURE_HZ)
    base = evaluate(env, lambda o: env.action_space.sample(), EVAL_SEEDS, EVAL_REPEATS, "случайная")

    env._cmd("CONFIG capture %d" % CAPTURE_HZ)
    # награда имеет тяжёлый хвост (редкие длинные комбо дают в десятки раз больше обычного),
    # поэтому нормализуем её; наблюдения уже приведены к разумному масштабу в плагине
    # monitor.csv с каждым эпизодом пишется в run_dir (папка уже создана выше)
    base_venv = DummyVecEnv([lambda: Monitor(env, filename=run_dir)])
    if VECNORM_FROM:
        venv = VecNormalize.load(VECNORM_FROM, base_venv)
        venv.training = True
        print("Статистика VecNormalize загружена из", VECNORM_FROM)
    else:
        print("VecNormalize: создаю новую статистику (нормализуется только награда)")
        venv = VecNormalize(base_venv, norm_obs=False, norm_reward=True,
                            clip_reward=10.0, gamma=GAMMA)

    if RESUME_FROM:
        model = PPO.load(RESUME_FROM, env=venv, device="cpu")
        print("Продолжаем обучение с", RESUME_FROM)
    else:
        model = PPO(
            "MlpPolicy", venv,
            n_steps=1024, batch_size=256, n_epochs=8,
            gamma=GAMMA, gae_lambda=0.95, learning_rate=3e-4,
            ent_coef=0.01, clip_range=0.2,
            policy_kwargs=dict(net_arch=[128, 128]),
            device="cpu", verbose=1,
        )

    print("\n--- Обучение: %d решений, режим %d Гц ---" % (TOTAL_DECISIONS, CAPTURE_HZ))
    model.learn(
        total_timesteps=TOTAL_DECISIONS,
        callback=[ScoreLogger(every=10),
                  CheckpointCallback(save_freq=10_000, save_path=run_dir, name_prefix="ppo",
                                     save_vecnormalize=True)],
        reset_num_timesteps=RESUME_FROM is None,
    )
    model.save(os.path.join(run_dir, "final"))
    venv.save(os.path.join(run_dir, "vecnormalize.pkl"))
    print("Модель сохранена в", run_dir)

    print("\n--- Проверка на контрольных картах (режим %d Гц, детерминированная политика) ---" % EVAL_CAPTURE_HZ)
    env._cmd("CONFIG capture %d" % EVAL_CAPTURE_HZ)
    trained = evaluate(env, lambda o: model.predict(o, deterministic=True)[0], EVAL_SEEDS, EVAL_REPEATS, "PPO")
    print("\nИтог: случайная %.0f -> PPO %.0f" % (base, trained))
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n!!! Ошибка:")
        traceback.print_exc()
    input("\nНажми Enter, чтобы закрыть окно...")
