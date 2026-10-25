using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Text;
using System.Threading;
using BepInEx;
using BepInEx.Configuration;
using HarmonyLib;
using UnityEngine;

namespace AgentProbe
{
    /// <summary>
    /// v0.2: окружение для RL внутри игры.
    ///  - TCP-сервер на localhost (текстовый протокол, по строке на команду);
    ///  - lockstep: игра стоит в FixedUpdate, пока Python не пришлёт следующую команду;
    ///  - RESET (со сменой сида карты), STEP (действие на K физических шагов);
    ///  - в обычном режиме (Python не подключён) игра работает как раньше + хоткеи для замеров.
    /// Хоткеи: F5=1x F6=2x F7=5x F8=10x F9=20x, F10=снять vSync, F11=captureFramerate вкл/выкл, Insert=рендер вкл/выкл.
    /// </summary>
    [BepInPlugin("dmitry.mdd.superflight.ai", "Superflight AI", "0.2.5")]
    public class Plugin : BaseUnityPlugin
    {
        public static Plugin Instance;
        private const int ObsDim = 40;
        private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

        // ---------- конфиг ----------
        private ConfigEntry<bool> _blockSaves;
        private ConfigEntry<int> _logEvery;
        private ConfigEntry<int> _port;
        private ConfigEntry<int> _frameSkipCfg;
        private ConfigEntry<float> _rayRangeCfg;
        private ConfigEntry<bool> _useCaptureCfg;
        private ConfigEntry<bool> _muteCfg;
        private ConfigEntry<bool> _noRenderCfg;

        // параметры, которые Python может менять командой CONFIG
        private int _frameSkip = 4;
        private float _rayRange = 1000f;
        private bool _useCapture = true;
        private bool _mute = true;
        private bool _noRender;

        // ---------- рефлексия по типам игры ----------
        private bool _reflectionReady, _reflectionFailed;
        private PropertyInfo _pLgm, _pPm, _pPp;
        private FieldInfo _fState, _fRun, _fCombo, _fSpeed, _fRb, _fProx, _fNearest, _fSurround;

        private bool _agentReflectionReady, _agentReflectionFailed;
        private FieldInfo _fPlayer, _fHor, _fVer, _fInputSens;
        private MethodInfo _mResetPlayer, _mLoadFlight, _mGenerate;
        private PropertyInfo _pLevelGen, _pSettings;
        private Type _tWorld, _tSettings;

        // ---------- режим агента ----------
        private volatile bool _agent;
        private TcpListener _listener;
        private Thread _acceptThread;
        private volatile TcpClient _pendingClient;
        private TcpClient _client;
        private NetworkStream _stream;
        private readonly byte[] _rbuf = new byte[8192];
        private int _rlen;

        private int _stepsLeft;
        private float _actH, _actV;
        private float _prevTotal;
        private int _epSteps;
        private bool _haveSeed;
        private int _currentSeed;
        private float _xSens = 1f, _ySens = 1f;
        private bool _xInv, _yInv;
        private int _decisions;
        private string _deferredReset;   // RESET, пришедший в FixedUpdate: выполняется в Update следующего кадра
        private int _resetAttempts;
        private int _resetPhase;          // 0 нет; 1 нужен сброс (делает Update); 2 ждём ровно 2 физ. тика после сброса
        private int _obsStep;             // номер физ. тика, на котором отдаём первый obs после сброса
        private int _tick;                // монотонный счётчик физических тиков (не обнуляется)
        private int _captureHz;           // 0 = 1/fixedDeltaTime
        private int _activePort;

        // восстановление настроек при отключении агента
        private int _savedVsync, _savedTargetFps, _savedCapture;
        private float _savedVolume, _savedTimeScale, _savedMaxDt;

        // ---------- ручные хоткеи / замеры ----------
        private float _targetScale = 1f;
        private bool _uncapped, _origSaved;
        private int _origVsync, _origTargetFps;
        private bool _renderOff;
        private readonly List<Camera> _maskCams = new List<Camera>();
        private readonly List<int> _masks = new List<int>();

        private int _frames, _fixedSteps, _stepCounter, _lastState = -1;
        private float _realT0, _gameT0;
        private int _decisionsAtReport;

        private static Vector3[] _rayDirs;
        private readonly HashSet<string> _loggedOnce = new HashSet<string>();

        // =====================================================================
        private void Awake()
        {
            Instance = this;

            _blockSaves = Config.Bind("General", "BlockSaveAndLeaderboard", true,
                "Глушить CollectPoints и вызовы Steam, чтобы бот не писал в сохранение и лидерборд.");
            _logEvery = Config.Bind("General", "LogEveryNFixedSteps", 50,
                "Как часто (в физ. шагах) писать [obs] в лог в обычном режиме.");
            _port = Config.Bind("Agent", "Port", 5555, "TCP-порт сервера на 127.0.0.1.");
            _frameSkipCfg = Config.Bind("Agent", "FrameSkip", 4, "Физических шагов на одно решение агента.");
            _rayRangeCfg = Config.Bind("Agent", "RayRange", 1000f, "Дальность лучей (ед. игры).");
            _useCaptureCfg = Config.Bind("Agent", "UseCaptureFramerate", true,
                "Фиксировать игровой dt кадра (1 кадр = 1 физ. шаг), чтобы счёт не зависел от скорости симуляции.");
            _muteCfg = Config.Bind("Agent", "MuteAudio", true, "Глушить звук, пока подключён агент.");
            _noRenderCfg = Config.Bind("Agent", "DisableRendering", false, "Отключать камеры, пока подключён агент.");

            _frameSkip = Math.Max(1, _frameSkipCfg.Value);
            _rayRange = Math.Max(10f, _rayRangeCfg.Value);
            _useCapture = _useCaptureCfg.Value;
            _mute = _muteCfg.Value;
            _noRender = _noRenderCfg.Value;

            Application.runInBackground = true;

            Harmony h = new Harmony("dmitry.mdd.superflight.ai");
            if (_blockSaves.Value)
            {
                Block(h, "LocalGameManager", "CollectPoints");
                Block(h, "SteamIntegration", "SubmitHighscore");
                Block(h, "SteamIntegration", "SubmitCombo");
                Block(h, "SteamIntegration", "UnlockAchievement");
                Block(h, "SteamIntegration", "IncrementStat");
            }
            PatchInput(h);

            _realT0 = Time.realtimeSinceStartup;
            _gameT0 = Time.time;

            // порт можно переопределить аргументом запуска игры: -aiport 5556 (для нескольких копий игры)
            _activePort = _port.Value;
            try
            {
                string[] args = Environment.GetCommandLineArgs();
                for (int i = 0; i < args.Length - 1; i++)
                {
                    int pp;
                    if (args[i] == "-aiport" && int.TryParse(args[i + 1], out pp)) _activePort = pp;
                }
            }
            catch (Exception) { }

            StartServer();
            Logger.LogInfo("Superflight AI 0.2.5 loaded. Port " + _activePort +
                           ". Hotkeys: F5-F9 timeScale, F10 uncap, F11 captureFramerate, Insert rendering");
        }

        private void OnDestroy() { StopServer(); }
        private void OnApplicationQuit() { StopAgent(); StopServer(); }

        // =====================================================================
        // Harmony
        // =====================================================================
        private static bool SkipOriginal() { return false; }

        private void Block(Harmony h, string typeName, string methodName)
        {
            try
            {
                Type t = AccessTools.TypeByName(typeName);
                if (t == null) { Logger.LogWarning("block: type not found: " + typeName); return; }
                MethodInfo m = AccessTools.Method(t, methodName);
                if (m == null) { Logger.LogWarning("block: method not found: " + typeName + "." + methodName); return; }
                MethodInfo prefix = typeof(Plugin).GetMethod("SkipOriginal", BindingFlags.Static | BindingFlags.NonPublic);
                h.Patch(m, new HarmonyMethod(prefix));
                Logger.LogInfo("blocked " + typeName + "." + methodName);
            }
            catch (Exception e)
            {
                Logger.LogWarning("block failed for " + typeName + "." + methodName + ": " + e.Message);
            }
        }

        private void PatchInput(Harmony h)
        {
            try
            {
                Type tPm = AccessTools.TypeByName("PlayerMovement");
                MethodInfo m = tPm != null ? AccessTools.Method(tPm, "GetInput") : null;
                if (m == null) { Logger.LogWarning("PlayerMovement.GetInput not found, agent input will not work"); return; }
                MethodInfo post = typeof(Plugin).GetMethod("GetInputPostfix", BindingFlags.Static | BindingFlags.NonPublic);
                h.Patch(m, null, new HarmonyMethod(post));
                Logger.LogInfo("patched PlayerMovement.GetInput (postfix)");
            }
            catch (Exception e)
            {
                Logger.LogWarning("PatchInput failed: " + e.Message);
            }
        }

        // выполняется после родного GetInput: перезаписываем поля осей значениями агента
        private static void GetInputPostfix(object __instance)
        {
            Plugin p = Instance;
            if (p == null || !p._agent) return;
            try { p.ApplyAgentInput(__instance); }
            catch (Exception e) { p.LogOnce("input", "ApplyAgentInput: " + e.Message); }
        }

        private void ApplyAgentInput(object pm)
        {
            if (_fInputSens == null || _fHor == null || _fVer == null) return;
            float sens = ToF(_fInputSens.GetValue(pm));
            float ah = _actH, av = _actV;
            // как в игре: диагональ нормализуется через |cos(pi*other/4)|
            float hx = ah * Mathf.Abs(Mathf.Cos(Mathf.PI * av / 4f));
            float vy = av * Mathf.Abs(Mathf.Cos(Mathf.PI * ah / 4f));
            float h = hx * sens * _xSens * (_xInv ? -1f : 1f);
            float v = vy * sens * _ySens * (_yInv ? -1f : 1f);
            _fHor.SetValue(pm, h);
            _fVer.SetValue(pm, v);
        }

        // =====================================================================
        // Рефлексия
        // =====================================================================
        private bool EnsureReflection()
        {
            if (_reflectionReady) return true;
            if (_reflectionFailed) return false;

            Type tLgm = AccessTools.TypeByName("LocalGameManager");
            Type tPm = AccessTools.TypeByName("PlayerMovement");
            Type tPp = AccessTools.TypeByName("PlayerProximeter");
            if (tLgm == null || tPm == null || tPp == null) return false;

            _pLgm = AccessTools.Property(tLgm, "Singleton");
            _pPm = AccessTools.Property(tPm, "Singleton");
            _pPp = AccessTools.Property(tPp, "Singleton");
            _fState = AccessTools.Field(tLgm, "playerState");
            _fRun = AccessTools.Field(tLgm, "scoreThisRun");
            _fCombo = AccessTools.Field(tLgm, "scoreThisCombo");
            _fSpeed = AccessTools.Field(tPm, "currentSpeed");
            _fRb = AccessTools.Field(tPm, "rb");
            _fProx = AccessTools.Field(tPp, "proximityAmount");
            _fNearest = AccessTools.Field(tPp, "distanceToNearestObstecle");
            _fSurround = AccessTools.Field(tPp, "numberOfObsteclesInTheSurrondingInPrecent");

            string missing = "";
            if (_pLgm == null) missing += " LocalGameManager.Singleton";
            if (_pPm == null) missing += " PlayerMovement.Singleton";
            if (_pPp == null) missing += " PlayerProximeter.Singleton";
            if (_fState == null) missing += " playerState";
            if (_fRun == null) missing += " scoreThisRun";
            if (_fCombo == null) missing += " scoreThisCombo";
            if (_fSpeed == null) missing += " currentSpeed";
            if (_fRb == null) missing += " rb";
            if (_fProx == null) missing += " proximityAmount";
            if (_fNearest == null) missing += " distanceToNearestObstecle";
            if (_fSurround == null) missing += " numberOfObsteclesInTheSurrondingInPrecent";
            if (missing.Length > 0)
            {
                _reflectionFailed = true;
                Logger.LogError("reflection failed, missing:" + missing);
                return false;
            }
            _reflectionReady = true;
            Logger.LogInfo("reflection OK");
            return true;
        }

        // то, что нужно только агенту: сброс, смена карты, поля осей
        private bool EnsureAgentReflection(out string error)
        {
            error = "";
            if (!EnsureReflection()) { error = "core reflection not ready (is the game in flight scene?)"; return false; }
            if (_agentReflectionReady) return true;
            if (_agentReflectionFailed) { error = "agent reflection failed earlier, see log"; return false; }

            Type tLgm = AccessTools.TypeByName("LocalGameManager");
            Type tPm = AccessTools.TypeByName("PlayerMovement");
            Type tLg = AccessTools.TypeByName("LevelGenerator");
            _tWorld = AccessTools.TypeByName("World");
            _tSettings = AccessTools.TypeByName("SettingsManager");

            _fPlayer = AccessTools.Field(tLgm, "player");
            _mResetPlayer = AccessTools.Method(tLgm, "ResetPlayer");
            _mLoadFlight = AccessTools.Method(tLgm, "LoadFlight");
            _fHor = AccessTools.Field(tPm, "horizontalInput");
            _fVer = AccessTools.Field(tPm, "verticalInput");
            _fInputSens = AccessTools.Field(tPm, "inputSensitivity");
            _pLevelGen = tLg != null ? AccessTools.Property(tLg, "Singleton") : null;
            _mGenerate = tLg != null ? AccessTools.Method(tLg, "GenerateLevel") : null;
            _pSettings = _tSettings != null ? AccessTools.Property(_tSettings, "Singleton") : null;

            string missing = "";
            if (_fPlayer == null) missing += " LocalGameManager.player";
            if (_mResetPlayer == null) missing += " ResetPlayer";
            if (_fHor == null) missing += " horizontalInput";
            if (_fVer == null) missing += " verticalInput";
            if (_fInputSens == null) missing += " inputSensitivity";
            if (missing.Length > 0)
            {
                _agentReflectionFailed = true;
                error = "missing:" + missing;
                Logger.LogError("agent reflection failed, " + error);
                return false;
            }
            if (_mLoadFlight == null) Logger.LogWarning("LoadFlight not found: RESET from menu will not work");
            if (_mGenerate == null || _tWorld == null || _pLevelGen == null)
                Logger.LogWarning("level generator / World not found: map seed switching disabled");

            _agentReflectionReady = true;
            return true;
        }

        private static object GetStatic(PropertyInfo p)
        {
            if (p == null) return null;
            try { return p.GetValue(null, null); }
            catch (Exception) { return null; }
        }

        private static float ToF(object o)
        {
            try { return Convert.ToSingle(o); }
            catch (Exception) { return float.NaN; }
        }

        private static float C(float x, float lim)
        {
            if (float.IsNaN(x) || float.IsInfinity(x)) return 0f;
            return Mathf.Clamp(x, -lim, lim);
        }

        private void LogOnce(string key, string msg)
        {
            if (_loggedOnce.Contains(key)) return;
            _loggedOnce.Add(key);
            Logger.LogWarning(msg);
        }

        private int ReadState()
        {
            object lgm = GetStatic(_pLgm);
            if (lgm == null || _fState == null) return 4;
            try { return Convert.ToInt32(_fState.GetValue(lgm)); }
            catch (Exception) { return 4; }
        }

        private float ReadSettingF(string name, float def)
        {
            try
            {
                if (_tSettings == null) return def;
                FieldInfo f = AccessTools.Field(_tSettings, name);
                if (f == null) return def;
                object src = f.IsStatic ? null : GetStatic(_pSettings);
                return Convert.ToSingle(f.GetValue(src));
            }
            catch (Exception) { return def; }
        }

        private bool ReadSettingB(string name, bool def)
        {
            try
            {
                if (_tSettings == null) return def;
                FieldInfo f = AccessTools.Field(_tSettings, name);
                if (f == null) return def;
                object src = f.IsStatic ? null : GetStatic(_pSettings);
                return Convert.ToBoolean(f.GetValue(src));
            }
            catch (Exception) { return def; }
        }

        private void RefreshInputScale()
        {
            _xSens = ReadSettingF("xAxisSensibility", 1f);
            _ySens = ReadSettingF("yAxisSensibility", 1f);
            _xInv = ReadSettingB("xAxisInverted", false);
            _yInv = ReadSettingB("yAxisInverted", false);
        }

        // =====================================================================
        // TCP-сервер
        // =====================================================================
        private void StartServer()
        {
            try
            {
                _listener = new TcpListener(IPAddress.Loopback, _activePort);
                _listener.Start();
                _acceptThread = new Thread(AcceptLoop);
                _acceptThread.IsBackground = true;
                _acceptThread.Start();
            }
            catch (Exception e)
            {
                Logger.LogError("cannot start server on port " + _activePort + ": " + e.Message);
            }
        }

        private void StopServer()
        {
            try { if (_listener != null) _listener.Stop(); } catch (Exception) { }
            _listener = null;
        }

        private void AcceptLoop()
        {
            while (_listener != null)
            {
                try
                {
                    TcpClient c = _listener.AcceptTcpClient();
                    if (_agent || _pendingClient != null) { c.Close(); continue; }
                    _pendingClient = c;
                }
                catch (Exception) { return; }
            }
        }

        private void Send(string line)
        {
            byte[] b = Encoding.ASCII.GetBytes(line + "\n");
            _stream.Write(b, 0, b.Length);
            _stream.Flush();
        }

        private string ReadLine()
        {
            while (true)
            {
                int idx = Array.IndexOf(_rbuf, (byte)'\n', 0, _rlen);
                if (idx >= 0)
                {
                    string s = Encoding.ASCII.GetString(_rbuf, 0, idx).TrimEnd('\r');
                    int rest = _rlen - idx - 1;
                    if (rest > 0) Buffer.BlockCopy(_rbuf, idx + 1, _rbuf, 0, rest);
                    _rlen = rest;
                    return s;
                }
                if (_rlen >= _rbuf.Length) throw new IOException("line too long");
                int n = _stream.Read(_rbuf, _rlen, _rbuf.Length - _rlen);
                if (n <= 0) throw new IOException("client closed connection");
                _rlen += n;
            }
        }

        private void StartAgent(TcpClient c)
        {
            _client = c;
            _client.NoDelay = true;
            _client.ReceiveTimeout = 60000;
            _client.SendTimeout = 10000;
            _stream = _client.GetStream();
            _rlen = 0;
            _stepsLeft = 0;
            _actH = _actV = 0f;
            _prevTotal = 0f;
            _epSteps = 0;
            _decisions = 0;

            _savedVsync = QualitySettings.vSyncCount;
            _savedTargetFps = Application.targetFrameRate;
            _savedCapture = Time.captureFramerate;
            _savedVolume = AudioListener.volume;
            _savedTimeScale = Time.timeScale;
            _savedMaxDt = Time.maximumDeltaTime;

            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = -1;
            ApplyCapture();
            if (_mute) AudioListener.volume = 0f;
            if (_noRender) SetRendering(false);

            _agent = true;
            Logger.LogInfo("[agent] client connected");
        }

        private void ApplyCapture()
        {
            if (_useCapture)
            {
                int cf = _captureHz > 1 ? _captureHz : Mathf.RoundToInt(1f / Time.fixedDeltaTime);
                Time.captureFramerate = cf;
                Time.timeScale = 1f;
            }
            else
            {
                Time.captureFramerate = 0;
            }
        }

        private void StopAgent()
        {
            bool was = _agent;
            _agent = false;
            _deferredReset = null;
            _resetPhase = 0;
            try { if (_stream != null) _stream.Close(); } catch (Exception) { }
            try { if (_client != null) _client.Close(); } catch (Exception) { }
            _stream = null;
            _client = null;
            if (!was) return;

            QualitySettings.vSyncCount = _savedVsync;
            Application.targetFrameRate = _savedTargetFps;
            Time.captureFramerate = _savedCapture;
            AudioListener.volume = _savedVolume;
            Time.timeScale = _savedTimeScale;
            Time.maximumDeltaTime = _savedMaxDt;
            SetRendering(true);
            Logger.LogInfo("[agent] disconnected, settings restored");
        }

        // «Выключить рендер» = cullingMask 0. Камеры остаются включёнными, Camera.main работает
        // (на него опирается много кода игры: цвета, облака, скайбокс).
        private void SetRendering(bool on)
        {
            try
            {
                if (on)
                {
                    for (int i = 0; i < _maskCams.Count; i++)
                        if (_maskCams[i] != null) _maskCams[i].cullingMask = _masks[i];
                    _maskCams.Clear();
                    _masks.Clear();
                    _renderOff = false;
                }
                else
                {
                    Camera[] cams = Camera.allCameras;
                    for (int i = 0; i < cams.Length; i++)
                    {
                        Camera c = cams[i];
                        if (c == null || _maskCams.Contains(c)) continue;
                        _maskCams.Add(c);
                        _masks.Add(c.cullingMask);
                        c.cullingMask = 0;
                    }
                    _renderOff = true;
                }
            }
            catch (Exception e) { Logger.LogWarning("SetRendering: " + e.Message); }
        }

        // =====================================================================
        // Команды от Python
        // =====================================================================
        // true => выйти из блокирующего цикла (пришёл STEP или QUIT)
        private bool HandleCommand(string line)
        {
            if (string.IsNullOrEmpty(line)) return false;
            string[] p = line.Split(' ');
            string cmd = p[0].ToUpperInvariant();

            if (cmd == "HELLO")
            {
                string err;
                if (!EnsureAgentReflection(out err)) { Send("ERR " + err); return false; }
                RefreshInputScale();
                Logger.LogInfo(string.Format(Inv,
                    "[agent] input scale: xSens={0} ySens={1} xInv={2} yInv={3}", _xSens, _ySens, _xInv, _yInv));
                Send("HELLO 2 " + ObsDim + " " + _frameSkip + " " + Time.fixedDeltaTime.ToString("R", Inv));
                return false;
            }
            if (cmd == "CONFIG" && p.Length >= 3)
            {
                ApplyConfig(p[1].ToLowerInvariant(), p[2]);
                Send("OK");
                return false;
            }
            if (cmd == "RESET")
            {
                string err;
                if (!EnsureAgentReflection(out err)) { Send("ERR " + err); return false; }
                _deferredReset = p.Length > 1 ? p[1] : "keep";
                _resetPhase = 1;
                _resetAttempts = 0;
                return true; // сброс выполнится в Update, как в самой игре; ответ уйдёт на кадр позже
            }
            if (cmd == "STEP" && p.Length >= 3)
            {
                _actH = Mathf.Clamp(float.Parse(p[1], NumberStyles.Float, Inv), -1f, 1f);
                _actV = Mathf.Clamp(float.Parse(p[2], NumberStyles.Float, Inv), -1f, 1f);
                _stepsLeft = _frameSkip;
                return true;
            }
            if (cmd == "QUIT")
            {
                StopAgent();
                return true;
            }
            Send("ERR unknown command: " + line);
            return false;
        }

        private void ApplyConfig(string key, string val)
        {
            try
            {
                if (key == "frameskip") _frameSkip = Math.Max(1, int.Parse(val, Inv));
                else if (key == "ray_range") _rayRange = Math.Max(10f, float.Parse(val, NumberStyles.Float, Inv));
                else if (key == "capture") { int hz = int.Parse(val, Inv); _useCapture = hz != 0; _captureHz = hz > 1 ? hz : 0; ApplyCapture(); }
                else if (key == "mute") { _mute = val == "1"; AudioListener.volume = _mute ? 0f : _savedVolume; }
                else if (key == "render") { if (val == "1") SetRendering(true); else SetRendering(false); }
                else if (key == "timescale") { _useCapture = false; ApplyCapture(); Time.timeScale = float.Parse(val, NumberStyles.Float, Inv); Time.maximumDeltaTime = 1f; }
                else Logger.LogWarning("[agent] unknown CONFIG key: " + key);
            }
            catch (Exception e) { Logger.LogWarning("[agent] CONFIG error: " + e.Message); }
        }

        // =====================================================================
        // Сброс эпизода и смена карты
        // =====================================================================
        private GameObject GetPlayerGO(object lgm)
        {
            try
            {
                object o = _fPlayer.GetValue(lgm);
                GameObject go = o as GameObject;
                if (go == null)
                {
                    Component c = o as Component;
                    if (c != null) go = c.gameObject;
                }
                return go;
            }
            catch (Exception) { return null; }
        }

        private void DoReset(string seedToken)
        {
            object lgm = GetStatic(_pLgm);
            if (lgm == null) { Logger.LogWarning("[agent] RESET: no LocalGameManager"); return; }

            bool changeSeed = false;
            int seed = 0;
            if (seedToken != "keep" && int.TryParse(seedToken, NumberStyles.Integer, Inv, out seed))
                changeSeed = !_haveSeed || seed != _currentSeed;

            if (changeSeed) GenerateLevelWithSeed(seed);

            GameObject old = GetPlayerGO(lgm);
            int state = ReadState();
            if (state == 4 && _mLoadFlight != null) _mLoadFlight.Invoke(lgm, null);
            else _mResetPlayer.Invoke(lgm, null);

            // старый игрок живёт до конца кадра: выключаем его сразу, чтобы не ловить ложные столкновения
            try { if (old != null) old.SetActive(false); } catch (Exception) { }
            // отложенный показ окна конца забега (RunEnded -> Invoke) в новом эпизоде не нужен
            try { ((MonoBehaviour)lgm).CancelInvoke("ActivateRunEndCanvas"); } catch (Exception) { }

            SyncPhysics();
            RefreshInputScale();
            if (_renderOff) SetRendering(false); // подхватить новые камеры, если появились

            _stepsLeft = 0;
            _actH = _actV = 0f;
            _prevTotal = 0f;
            _epSteps = 0;
        }

        private void GenerateLevelWithSeed(int seed)
        {
            if (_mGenerate == null || _tWorld == null || _pLevelGen == null)
            {
                LogOnce("noseed", "[agent] seed switching not available");
                return;
            }
            try
            {
                System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
                object world = Activator.CreateInstance(_tWorld, new object[] { seed });
                object lg = GetStatic(_pLevelGen);
                _mGenerate.Invoke(lg, new object[] { world });
                sw.Stop();
                _currentSeed = seed;
                _haveSeed = true;
                Logger.LogInfo(string.Format(Inv, "[agent] GenerateLevel seed={0} took {1} ms", seed, sw.ElapsedMilliseconds));
            }
            catch (Exception e)
            {
                Logger.LogWarning("[agent] GenerateLevel failed: " + e.Message);
            }
        }

        private static void SyncPhysics()
        {
            try
            {
                MethodInfo m = typeof(Physics).GetMethod("SyncTransforms", BindingFlags.Static | BindingFlags.Public);
                if (m != null) m.Invoke(null, null);
            }
            catch (Exception) { }
        }

        // =====================================================================
        // Наблюдения и награда
        // =====================================================================
        private static Vector3[] RayDirs()
        {
            if (_rayDirs != null) return _rayDirs;
            List<Vector3> d = new List<Vector3>();
            d.Add(Vector3.forward);
            float[] thetas = new float[] { 12f, 30f, 55f };
            for (int r = 0; r < thetas.Length; r++)
            {
                float th = thetas[r] * Mathf.Deg2Rad;
                for (int k = 0; k < 8; k++)
                {
                    float phi = k * 45f * Mathf.Deg2Rad;
                    d.Add(new Vector3(Mathf.Sin(th) * Mathf.Cos(phi), Mathf.Sin(th) * Mathf.Sin(phi), Mathf.Cos(th)));
                }
            }
            _rayDirs = d.ToArray();
            return _rayDirs;
        }

        private float CastRay(Vector3 origin, Vector3 dir, Transform playerRoot)
        {
            float best = _rayRange;
            RaycastHit[] hits = Physics.RaycastAll(origin, dir, _rayRange, -1, QueryTriggerInteraction.Ignore);
            for (int i = 0; i < hits.Length; i++)
            {
                Collider col = hits[i].collider;
                if (col == null) continue;
                if (col.transform.root == playerRoot) continue;
                if (hits[i].distance < best) best = hits[i].distance;
            }
            return best / _rayRange;
        }

        private float[] BuildObs()
        {
            float[] o = new float[ObsDim];
            object pm = GetStatic(_pPm);
            object pp = GetStatic(_pPp);
            if (pm == null || pp == null) return o;
            Rigidbody rb = _fRb.GetValue(pm) as Rigidbody;
            if (rb == null) return o;

            Transform t = rb.transform;
            Vector3 vloc = t.InverseTransformDirection(rb.velocity);
            Vector3 grav = t.InverseTransformDirection(Vector3.down);
            Vector3 wloc = t.InverseTransformDirection(rb.angularVelocity);

            int i = 0;
            o[i++] = C(ToF(_fSpeed.GetValue(pm)) / 1250f, 2f);
            o[i++] = C(vloc.x / 250f, 3f);
            o[i++] = C(vloc.y / 250f, 3f);
            o[i++] = C(vloc.z / 250f, 3f);
            o[i++] = C(grav.x, 1f);
            o[i++] = C(grav.y, 1f);
            o[i++] = C(grav.z, 1f);
            o[i++] = C(wloc.x / 5f, 3f);
            o[i++] = C(wloc.y / 5f, 3f);
            o[i++] = C(wloc.z / 5f, 3f);
            o[i++] = C(ToF(_fProx.GetValue(pp)), 2f);
            o[i++] = C(ToF(_fNearest.GetValue(pp)), 2f);
            o[i++] = C(Mathf.Log(1f + Mathf.Max(0f, ToF(_fSurround.GetValue(pp)))) / 5f, 2f);
            o[i++] = _actH;
            o[i++] = _actV;

            Vector3[] dirs = RayDirs();
            Vector3 origin = rb.position;
            Transform root = t.root;
            for (int k = 0; k < dirs.Length && i < ObsDim; k++)
                o[i++] = CastRay(origin, t.TransformDirection(dirs[k]), root);
            return o;
        }

        private void SendObs(bool done)
        {
            object lgm = GetStatic(_pLgm);
            float run = lgm != null ? ToF(_fRun.GetValue(lgm)) : 0f;
            float combo = lgm != null ? ToF(_fCombo.GetValue(lgm)) : 0f;
            if (float.IsNaN(run)) run = 0f;
            if (float.IsNaN(combo)) combo = 0f;
            float total = run + combo;
            float reward = total - _prevTotal;
            if (reward < 0f || float.IsNaN(reward)) reward = 0f;
            _prevTotal = total;

            float[] obs = BuildObs();
            StringBuilder sb = new StringBuilder(512);
            sb.Append("OBS ").Append(done ? 1 : 0);
            sb.Append(' ').Append(reward.ToString("R", Inv));
            sb.Append(' ').Append(total.ToString("R", Inv));
            sb.Append(' ').Append(Time.time.ToString("F2", Inv));
            sb.Append(' ').Append(_epSteps);
            for (int i = 0; i < obs.Length; i++)
                sb.Append(' ').Append(obs[i].ToString("0.####", Inv));
            Send(sb.ToString());
        }

        // =====================================================================
        // Основной цикл
        // =====================================================================
        private void Update()
        {
            _frames++;

            TcpClient pending = _pendingClient;
            if (pending != null && !_agent)
            {
                _pendingClient = null;
                StartAgent(pending);
            }

            if (_agent && _resetPhase == 1) ProcessDeferredReset();

            HandleHotkeys();
            if (!_agent) EnforceScale();
            ReportThroughput();
        }

        private void FixedUpdate()
        {
            _fixedSteps++;
            _tick++;
            if (_agent) { AgentTick(); return; }
            PassiveLog();
        }

        private void AgentTick()
        {
            if (_resetPhase == 1) return; // сброс сделает Update
            if (_resetPhase == 2)
            {
                // ровно 2 физических тика после сброса: у нового игрока к этому моменту точно отработал Start(),
                // а число тиков не зависит от того, сколько кадров Unity уложил между ними (детерминизм)
                if (_tick < _obsStep) return;
                _resetPhase = 0;
                _deferredReset = null;
                _resetAttempts = 0;
                try { SendObs(false); }
                catch (Exception e) { LogException("send obs after reset", e); StopAgent(); return; }
                CommandLoop(true);
                return;
            }
            try
            {
                if (_stepsLeft > 0)
                {
                    _stepsLeft--;
                    _epSteps++;
                    bool dead = ReadState() != 0;
                    if (_stepsLeft > 0 && !dead) return;
                    _decisions++;
                    _stepsLeft = 0;
                    SendObs(dead);
                }
            }
            catch (Exception e)
            {
                LogException("agent tick", e);
                StopAgent();
                return;
            }
            CommandLoop(true);
        }

        // блокирующий приём команд; fromFixed: true, если вызваны из FixedUpdate
        private void CommandLoop(bool fromFixed)
        {
            try
            {
                while (_agent)
                {
                    string line = ReadLine();
                    if (HandleCommand(line)) break;
                }
            }
            catch (Exception e)
            {
                LogException("agent loop", e);
                StopAgent();
            }
        }

        // Сброс ведёт Update (как в самой игре: клавиша / UI-кнопка), до 3 попыток по одной на кадр.
        // Ответ Python уходит позже, из AgentTick, ровно через 2 физических тика.
        private void ProcessDeferredReset()
        {
            try
            {
                DoReset(_deferredReset);
            }
            catch (Exception e)
            {
                _resetAttempts++;
                LogException("reset attempt " + _resetAttempts, e);
                if (_resetAttempts < 3) return;
                _resetPhase = 0;
                _deferredReset = null;
                try { Send("ERR reset failed: " + Short(e)); }
                catch (Exception) { StopAgent(); return; }
                CommandLoop(false);
                return;
            }
            _resetPhase = 2;
            _obsStep = _tick + 2;
        }

        private static Exception Unwrap(Exception e)
        {
            while (e is TargetInvocationException && e.InnerException != null) e = e.InnerException;
            return e;
        }

        private void LogException(string where, Exception e)
        {
            Exception b = Unwrap(e);
            Logger.LogWarning("[agent] " + where + ": " + b.GetType().Name + ": " + b.Message + "\n" + b.StackTrace);
        }

        private static string Short(Exception e)
        {
            Exception b = Unwrap(e);
            return (b.GetType().Name + ": " + b.Message).Replace('\n', ' ').Replace('\r', ' ');
        }

        // ---------- обычный режим ----------
        private void PassiveLog()
        {
            if (!EnsureReflection()) return;
            object lgm = GetStatic(_pLgm);
            if (lgm == null) return;

            int state;
            try { state = Convert.ToInt32(_fState.GetValue(lgm)); }
            catch (Exception) { return; }

            if (state != _lastState)
            {
                Logger.LogInfo(string.Format(Inv,
                    "[state] {0} -> {1}  runScore={2:F0} comboScore={3:F0} gameTime={4:F2}",
                    _lastState, state, ToF(_fRun.GetValue(lgm)), ToF(_fCombo.GetValue(lgm)), Time.time));
                _lastState = state;
            }
            if (state != 0) return;

            _stepCounter++;
            if (_stepCounter % Math.Max(1, _logEvery.Value) != 0) return;

            object pm = GetStatic(_pPm);
            object pp = GetStatic(_pPp);
            if (pm == null || pp == null) return;
            Rigidbody rb = _fRb.GetValue(pm) as Rigidbody;
            if (rb == null) return;
            Vector3 pos = rb.position;
            Vector3 vel = rb.velocity;

            Logger.LogInfo(string.Format(Inv,
                "[obs] t={0:F2} speed={1:F1} pos=({2:F0},{3:F0},{4:F0}) vel=({5:F1},{6:F1},{7:F1}) prox={8:F2} nearest={9:F2} surround={10:F2} run={11:F0} combo={12:F0}",
                Time.time, ToF(_fSpeed.GetValue(pm)),
                pos.x, pos.y, pos.z, vel.x, vel.y, vel.z,
                ToF(_fProx.GetValue(pp)), ToF(_fNearest.GetValue(pp)), ToF(_fSurround.GetValue(pp)),
                ToF(_fRun.GetValue(lgm)), ToF(_fCombo.GetValue(lgm))));
        }

        // ---------- хоткеи и замеры ----------
        private void HandleHotkeys()
        {
            if (_agent) return;
            if (Input.GetKeyDown(KeyCode.F5)) SetScale(1f);
            if (Input.GetKeyDown(KeyCode.F6)) SetScale(2f);
            if (Input.GetKeyDown(KeyCode.F7)) SetScale(5f);
            if (Input.GetKeyDown(KeyCode.F8)) SetScale(10f);
            if (Input.GetKeyDown(KeyCode.F9)) SetScale(20f);
            if (Input.GetKeyDown(KeyCode.F10)) ToggleUncap();
            if (Input.GetKeyDown(KeyCode.F11)) ToggleCapture();
            if (Input.GetKeyDown(KeyCode.Insert)) { SetRendering(_renderOff); Logger.LogInfo("rendering = " + (!_renderOff)); }
        }

        private void SetScale(float s)
        {
            _targetScale = s;
            Time.maximumDeltaTime = s > 1f ? 1f : 0.3333333f;
            if (Time.timeScale > 0.99f) Time.timeScale = s;
            Logger.LogInfo("target timeScale = " + s);
        }

        private void EnforceScale()
        {
            if (_targetScale <= 1f) return;
            if (Time.timeScale > 0.99f && Math.Abs(Time.timeScale - _targetScale) > 0.01f)
            {
                if (_lastState == 4) return;
                Time.timeScale = _targetScale;
            }
        }

        private void ToggleCapture()
        {
            if (Time.captureFramerate > 0)
            {
                Time.captureFramerate = 0;
                Logger.LogInfo("captureFramerate = off");
            }
            else
            {
                int cf = Mathf.RoundToInt(1f / Time.fixedDeltaTime);
                Time.captureFramerate = cf;
                Time.timeScale = 1f;
                _targetScale = 1f;
                Logger.LogInfo("captureFramerate = " + cf + " (1 frame = 1 physics step; use F10 to uncap vSync)");
            }
        }

        private void ToggleUncap()
        {
            if (!_origSaved)
            {
                _origVsync = QualitySettings.vSyncCount;
                _origTargetFps = Application.targetFrameRate;
                _origSaved = true;
                Logger.LogInfo("original vSyncCount=" + _origVsync + " targetFrameRate=" + _origTargetFps);
            }
            _uncapped = !_uncapped;
            QualitySettings.vSyncCount = _uncapped ? 0 : _origVsync;
            Application.targetFrameRate = _uncapped ? -1 : _origTargetFps;
            Logger.LogInfo("uncapped = " + _uncapped);
        }

        private delegate long BytesFn();

        // байты -> МБ; любой сбой (нет метода, Profiler недоступен) даёт "n/a", плагин не падает
        private static string MemMb(BytesFn f)
        {
            try { return (f() / 1048576L).ToString(Inv); }
            catch (Exception) { return "n/a"; }
        }

        private void ReportThroughput()
        {
            float realNow = Time.realtimeSinceStartup;
            float dr = realNow - _realT0;
            if (dr < 2f) return;
            float dg = Time.time - _gameT0;
            int dec = _decisions - _decisionsAtReport;
            Logger.LogInfo(string.Format(Inv,
                "[perf] mode={0} timeScale={1:F1} capture={2} fps={3:F0} physSteps/s={4:F0} decisions/s={5:F0} gameSec/realSec={6:F2} | mem MB: mono={7} unityAlloc={8} unityReserved={9} private={10} workingSet={11}",
                _agent ? "agent" : "manual", Time.timeScale, Time.captureFramerate,
                _frames / dr, _fixedSteps / dr, dec / dr, dg / dr,
                MemMb(delegate { return GC.GetTotalMemory(false); }),
                MemMb(delegate { return UnityEngine.Profiling.Profiler.GetTotalAllocatedMemoryLong(); }),
                MemMb(delegate { return UnityEngine.Profiling.Profiler.GetTotalReservedMemoryLong(); }),
                MemMb(delegate { return System.Diagnostics.Process.GetCurrentProcess().PrivateMemorySize64; }),
                MemMb(delegate { return System.Diagnostics.Process.GetCurrentProcess().WorkingSet64; })));
            _frames = 0;
            _fixedSteps = 0;
            _decisionsAtReport = _decisions;
            _realT0 = realNow;
            _gameT0 = Time.time;
        }
    }
}
