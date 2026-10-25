# Superflight RL

Обучение RL-агента (PPO, Stable-Baselines3) игре [Superflight](https://store.steampowered.com/app/732430/Superflight/).

*Reinforcement-learning agent (PPO / Stable-Baselines3) for the game Superflight. A BepInEx plugin exposes the game
over a local TCP socket in lockstep mode; a Gymnasium environment drives it from Python.*

## Как это устроено

- **`Plugin.cs`** — плагин BepInEx 5 для игры (Unity, Mono). Поднимает TCP-сервер `127.0.0.1:5555`,
  подменяет ввод игрока действиями агента, выполняет игру в lockstep (физика ждёт команду из Python),
  отдаёт наблюдение (40 чисел: скорость, ориентация, близость к скалам, 25 лучей) и награду.
  Отправка рекордов в Steam и сохранение очков отключаются, пока плагин установлен.
- **`superflight_env.py`** — Gymnasium-окружение поверх протокола плагина.
  Действие: `MultiDiscrete([3, 3])` — {-1, 0, +1} по горизонтали и вертикали.
- **`train_ppo.py`** — обучение PPO на одних картах (сидах) и проверка на других, сравнение со случайной стратегией.
- **`test_env.py`**, **`bench_capture.py`**, **`bench_determinism.py`** — проверка связки, замеры скорости и детерминизма.

## Требования

- Superflight (Steam, Windows x64)
- [BepInEx 5.4.x](https://github.com/BepInEx/BepInEx/releases) (x64), установленный в папку игры
- .NET SDK (для сборки плагина под `net35`)
- Python 3.10+

## Установка

1. Установить BepInEx в папку игры и один раз запустить игру (создадутся папки `BepInEx\...`).
2. Собрать плагин, указав папку игры (игра должна быть закрыта):
   ```
   dotnet build -p:GameDir="C:\path\to\steamapps\common\SuperFlight"
   ```
   DLL `Superflight.AI.dll` копируется в `BepInEx\plugins` автоматически.
3. Python-зависимости:
   ```
   pip install -r requirements.txt
   ```

## Запуск

1. Запустить игру (рекомендуется оконный режим 640x480, минимальная графика).
2. Проверить связь: `python test_env.py`
3. Обучение: `python train_ppo.py` (настройки — в блоке в начале файла). Результаты пишутся в `runs/`.

Лог плагина: `<папка игры>\BepInEx\LogOutput.log`.

## Протокол

По строке на команду:

| Команда | Ответ |
|---|---|
| `HELLO` | `HELLO <версия> <obs_dim> <frameskip> <fixedDt>` |
| `CONFIG <ключ> <знач>` (frameskip, ray_range, capture, mute, render, timescale) | `OK` |
| `RESET <seed\|keep>` | `OBS <done> <reward> <score> <t> <steps> <obs...>` |
| `STEP <h> <v>` | `OBS ...` |
| `QUIT` | — |

## Замечания

- Чтобы вернуть игре обычный режим (рекорды, Steam), удалите `Superflight.AI.dll` из `BepInEx\plugins`.
- Файлы игры в репозиторий не входят.
- Проект не связан с разработчиками Superflight.
