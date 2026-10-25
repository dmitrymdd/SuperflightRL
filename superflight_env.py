"""
Gymnasium-окружение для Superflight поверх плагина Superflight AI (BepInEx).

Протокол (текст, по строке на команду, localhost:5555):
  HELLO                 -> HELLO <версия> <размер_obs> <frameskip> <fixedDt>
  CONFIG <ключ> <знач>  -> OK    (ключи: frameskip, ray_range, capture, mute, render, timescale)
  RESET <seed|keep>     -> OBS 0 0 <score> <t> <steps> <obs...>
  STEP <h> <v>          -> OBS <done> <reward> <score> <t> <steps> <obs...>
  QUIT
"""
import socket
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces

ACTION_VALUES = (-1.0, 0.0, 1.0)


class SuperflightEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        host="127.0.0.1",
        port=5555,
        frame_skip=4,
        ray_range=1000.0,
        max_decisions=3000,
        map_seeds=None,
        reward_scale=0.01,
        connect_timeout=30.0,
        io_timeout=120.0,
        extra_config=None,
    ):
        super().__init__()
        self.max_decisions = max_decisions
        self.map_seeds = list(map_seeds) if map_seeds else None
        self.reward_scale = reward_scale
        self._decisions = 0
        self._score = 0.0

        self._sock = self._connect(host, port, connect_timeout)
        self._sock.settimeout(io_timeout)
        self._file = self._sock.makefile("rw", encoding="ascii", newline="\n")

        hello = self._cmd("HELLO").split()
        if hello[0] != "HELLO":
            raise RuntimeError("Плагин ответил ошибкой: " + " ".join(hello))
        self.obs_dim = int(hello[2])
        self.fixed_dt = float(hello[4])

        self._cmd("CONFIG frameskip %d" % frame_skip)
        self._cmd("CONFIG ray_range %s" % ray_range)
        for k, v in (extra_config or {}).items():
            self._cmd("CONFIG %s %s" % (k, v))
        self.frame_skip = frame_skip

        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), dtype=np.float32)
        # два независимых выбора из {-1, 0, +1}: горизонталь и вертикаль
        self.action_space = spaces.MultiDiscrete([3, 3])

    # ---------------- сеть ----------------
    @staticmethod
    def _connect(host, port, timeout):
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            try:
                s = socket.create_connection((host, port), timeout=5)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return s
            except OSError as e:
                last = e
                time.sleep(0.5)
        raise ConnectionError("Не удалось подключиться к игре на %s:%s (%s). "
                              "Игра запущена? Плагин загружен?" % (host, port, last))

    def _cmd(self, line):
        self._file.write(line + "\n")
        self._file.flush()
        resp = self._file.readline()
        if not resp:
            raise ConnectionError("Игра закрыла соединение")
        resp = resp.strip()
        if resp.startswith("ERR"):
            raise RuntimeError(resp)
        return resp

    def _parse_obs(self, resp):
        parts = resp.split()
        if parts[0] != "OBS":
            raise RuntimeError("Неожиданный ответ: " + resp[:80])
        done = parts[1] == "1"
        reward = float(parts[2])
        score = float(parts[3])
        game_time = float(parts[4])
        steps = int(parts[5])
        obs = np.asarray(parts[6:], dtype=np.float32)
        return obs, done, reward, score, game_time, steps

    # ---------------- Gymnasium API ----------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        map_seed = "keep"
        if options and "map_seed" in options:
            map_seed = str(int(options["map_seed"]))
        elif self.map_seeds:
            map_seed = str(int(self.np_random.choice(self.map_seeds)))

        resp = self._cmd("RESET " + map_seed)
        obs, _, _, score, game_time, _ = self._parse_obs(resp)
        self._decisions = 0
        self._score = score
        return obs, {"score": score, "game_time": game_time, "map_seed": map_seed}

    def step(self, action):
        a = np.asarray(action).reshape(-1)
        h = ACTION_VALUES[int(a[0])]
        v = ACTION_VALUES[int(a[1])]
        resp = self._cmd("STEP %s %s" % (h, v))
        obs, done, reward, score, game_time, steps = self._parse_obs(resp)
        self._decisions += 1
        self._score = score
        truncated = (not done) and self._decisions >= self.max_decisions
        info = {"score": score, "game_time": game_time, "physics_steps": steps}
        return obs, reward * self.reward_scale, done, truncated, info

    def close(self):
        try:
            self._file.write("QUIT\n")
            self._file.flush()
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass
