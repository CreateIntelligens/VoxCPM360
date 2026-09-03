import {
  AudioLines,
  BarChart3,
  BrainCircuit,
  Check,
  ChevronDown,
  CircleAlert,
  Clock3,
  Cpu,
  Dices,
  Download,
  FileAudio,
  Gauge,
  History,
  LoaderCircle,
  Mic2,
  RefreshCw,
  Server,
  SlidersHorizontal,
  Sparkles,
  Upload,
  X,
} from "lucide-react";
import {
  FormEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  fetchCatalog,
  fetchGenerationHistory,
  synthesize,
  synthesizeStream,
  type StreamingAudioChunk,
} from "./api";
import ModelComparison from "./ModelComparison";
import { usePersistentState } from "./usePersistentState";
import {
  REFERENCE_AUDIO_LANGUAGE_LABELS,
  type Catalog,
  type Engine,
  type HistoryItem,
  type ModelVersion,
  type ReferenceAudioPreset,
} from "./types";

const DEFAULT_TEXT = "逐家好，歡迎使用 VoxCPM 360 多模型語音工作室。";
const HISTORY_LIMIT = 100;

class StreamingAudioPlayer {
  private readonly context = new AudioContext();
  private readonly sources = new Set<AudioBufferSourceNode>();
  private scheduledAt = 0;
  private complete = false;
  private stopped = false;

  async prepare(): Promise<void> {
    if (this.context.state === "suspended") {
      await this.context.resume();
    }
    this.scheduledAt = this.context.currentTime + 0.08;
  }

  enqueue({ samples, sampleRate }: StreamingAudioChunk): void {
    if (this.stopped || samples.length === 0) {
      return;
    }
    const buffer = this.context.createBuffer(1, samples.length, sampleRate);
    buffer.copyToChannel(samples, 0);
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context.destination);
    source.onended = () => {
      this.sources.delete(source);
      this.closeWhenIdle();
    };
    this.sources.add(source);
    const startAt = Math.max(
      this.scheduledAt,
      this.context.currentTime + 0.04,
    );
    source.start(startAt);
    this.scheduledAt = startAt + buffer.duration;
  }

  finish(): void {
    this.complete = true;
    this.closeWhenIdle();
  }

  stop(): void {
    if (this.stopped) {
      return;
    }
    this.stopped = true;
    this.sources.forEach((source) => {
      try {
        source.stop();
      } catch {
        // 已自然播放完畢的 source 不需要再停止。
      }
    });
    this.sources.clear();
    void this.context.close();
  }

  private closeWhenIdle(): void {
    if (this.complete && this.sources.size === 0 && !this.stopped) {
      this.stopped = true;
      void this.context.close();
    }
  }
}

function createHistoryId(): string {
  if (typeof globalThis.crypto?.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function createRandomSeed(): string {
  if (typeof globalThis.crypto?.getRandomValues === "function") {
    const value = new Uint32Array(1);
    globalThis.crypto.getRandomValues(value);
    return String(value[0] & 0x7fffffff);
  }
  return String(Math.floor(Math.random() * 0x80000000));
}

function seedHelpText(streamingActive: boolean, seed: string): string {
  if (streamingActive) {
    return "串流端點目前不套用指定種子";
  }
  if (seed) {
    return `目前固定為 ${seed}；重骰會換一組結果`;
  }
  return "留空則每次隨機；重骰後可用同一種子重現結果";
}

function formatReferenceLabel(
  referenceAudio: File | undefined,
  preset: ReferenceAudioPreset | undefined,
): string {
  if (referenceAudio) {
    const sizeMb = (referenceAudio.size / 1024 / 1024).toFixed(2);
    return `自訂：${referenceAudio.name}（${sizeMb} MB）`;
  }
  if (preset) {
    return `${preset.label} · ${preset.description}`;
  }
  return "未指定參考音";
}

function App() {
  const [activeView, setActiveView] = usePersistentState<
    "studio" | "comparison"
  >(
    "active-view",
    "studio",
  );
  const [comparisonReloadKey, setComparisonReloadKey] = useState(0);
  const [catalog, setCatalog] = useState<Catalog>({
    engines: [],
    reference_presets: [],
    default_reference_preset_id: "",
  });
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [catalogError, setCatalogError] = useState("");
  const [engineId, setEngineId] = usePersistentState("engine-id", "");
  const [modelId, setModelId] = usePersistentState("model-id-v3", "");
  const [speakerId, setSpeakerId] = usePersistentState("speaker-id", "");
  const [text, setText] = usePersistentState("target-text", DEFAULT_TEXT);
  const [controlInstruction, setControlInstruction] = usePersistentState(
    "control-instruction",
    "",
  );
  const [promptText, setPromptText] = usePersistentState("prompt-text", "");
  const [referenceAudio, setReferenceAudio] = useState<File>();
  const [referencePresetId, setReferencePresetId] = usePersistentState(
    "reference-preset-id",
    "",
  );
  const [cfgValue, setCfgValue] = usePersistentState("cfg-value", 2);
  // 與 api.py 的 Form 預設對齊。steps=10 會讓 diffusion 沒收斂完就輸出
  // （實聽「亂叫、聽不懂」），normalize=false 則會讓音量爆掉；兩者原本都
  // 只寫在後端，但前端每次都顯式送值，後端預設等於被架空。
  // key 加 -v2 是為了讓已存舊值的瀏覽器重新套用新預設。
  const [steps, setSteps] = usePersistentState("inference-steps-v2", 30);
  const [speed, setSpeed] = usePersistentState("speech-speed", 1);
  const [normalize, setNormalize] = usePersistentState("normalize-v2", true);
  const [denoise, setDenoise] = usePersistentState("denoise", false);
  const [streaming, setStreaming] = usePersistentState("streaming", false);
  const [seed, setSeed] = usePersistentState("seed", "");
  const [advancedOpen, setAdvancedOpen] = usePersistentState(
    "advanced-open",
    false,
  );
  const [generating, setGenerating] = useState(false);
  const [generateError, setGenerateError] = useState("");
  const [currentHistoryId, setCurrentHistoryId] = useState("");
  const [autoPlayHistoryId, setAutoPlayHistoryId] = useState("");
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const audioUrlsRef = useRef(new Set<string>());
  const streamingPlayerRef = useRef<StreamingAudioPlayer | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const selectedEngine = useMemo(
    () => catalog.engines.find((engine) => engine.id === engineId),
    [catalog.engines, engineId],
  );
  const selectedModel = useMemo(
    () => selectedEngine?.models.find((model) => model.id === modelId),
    [selectedEngine, modelId],
  );
  const speakers = selectedModel?.speakers || [];
  const selectedSpeaker = speakers.find((speaker) => speaker.id === speakerId);
  const currentHistoryItem = useMemo(
    () => history.find((item) => item.id === currentHistoryId),
    [currentHistoryId, history],
  );
  const supportsControlInstruction = Boolean(
    selectedEngine?.capabilities.control_instruction,
  );
  const supportsPromptTranscript = Boolean(
    selectedEngine?.capabilities.prompt_transcript,
  );
  const supportsStreaming = Boolean(selectedEngine?.capabilities.streaming);
  const streamingActive = streaming && supportsStreaming;
  const effectiveSpeed = streamingActive ? 1 : speed;
  const selectedReferencePreset = useMemo(
    () =>
      catalog.reference_presets.find(
        (item) => item.id === referencePresetId,
      ),
    [catalog.reference_presets, referencePresetId],
  );
  const promptCloningActive = Boolean(
    promptText.trim() && (referenceAudio || selectedReferencePreset),
  );

  // 內建聲音的逐字稿是固定事實，直接填入並鎖為唯讀 —— 改了只會讓克隆變差。
  // 自訂上傳時清空並開放編輯：我們不知道使用者的音檔在講什麼，只能由他自己填。
  useEffect(() => {
    if (referenceAudio) {
      // 切到自訂上傳：清掉殘留的預設逐字稿，否則會拿別支音檔的內容去對，
      // 比留空更糟 —— 使用者還不見得會發現。
      setPromptText((current) => {
        const isPresetText = catalog.reference_presets.some(
          (item) => item.prompt_text && item.prompt_text === current.trim(),
        );
        return isPresetText ? "" : current;
      });
      return;
    }
    if (selectedReferencePreset?.prompt_text) {
      setPromptText(selectedReferencePreset.prompt_text);
    }
  }, [
    selectedReferencePreset,
    referenceAudio,
    catalog.reference_presets,
    setPromptText,
  ]);

  const loadGlobalHistory = useCallback(async () => {
    try {
      const records = await fetchGenerationHistory(HISTORY_LIMIT);
      setHistory((current) => {
        const localAudioUrls = new Map(
          current
            .filter((item) => audioUrlsRef.current.has(item.audioUrl))
            .map((item) => [item.id, item.audioUrl]),
        );
        const next = records.map((item) => ({
          ...item,
          audioUrl: localAudioUrls.get(item.id) || item.audioUrl,
        }));
        const nextIds = new Set(next.map((item) => item.id));
        current
          .filter((item) => !nextIds.has(item.id))
          .forEach((item) => {
            if (audioUrlsRef.current.delete(item.audioUrl)) {
              URL.revokeObjectURL(item.audioUrl);
            }
          });
        return next;
      });
      setCurrentHistoryId((currentId) =>
        records.some((item) => item.id === currentId)
          ? currentId
          : records[0]?.id || "",
      );
    } catch {
      // A temporary sync failure must not discard history already visible to the user.
    }
  }, []);
  let referencePresetState = "success";
  if (catalogLoading) {
    referencePresetState = "loading";
  } else if (catalog.reference_presets.length === 0) {
    referencePresetState = "error";
  } else if (referenceAudio) {
    referencePresetState = "custom";
  }

  const loadCatalog = async () => {
    setCatalogLoading(true);
    setCatalogError("");
    try {
      const next = await fetchCatalog();
      setCatalog(next);
      const nextEngine =
        next.engines.find((engine) => engine.id === engineId) ||
        next.engines.find((engine) => engine.online) ||
        next.engines[0];
      setEngineId(nextEngine?.id || "");
      const nextModel =
        nextEngine?.models.find((model) => model.id === modelId) ||
        nextEngine?.models.find((model) => model.loaded) ||
        nextEngine?.models.find((model) => model.id.includes("ft-mixed-lr2e5-avgE-e12run-0820")) ||
        nextEngine?.models.find((model) => model.online !== false) ||
        nextEngine?.models[0];
      setModelId(nextModel?.id || "");
      setSpeakerId((currentId) => {
        const nextSpeaker =
          nextModel?.speakers?.find((speaker) => speaker.id === currentId) ||
          nextModel?.speakers?.[0];
        return nextSpeaker?.id || "";
      });
      setReferencePresetId((currentId) => {
        const nextPreset =
          next.reference_presets.find((item) => item.id === currentId) ||
          next.reference_presets.find(
            (item) => item.id === next.default_reference_preset_id,
          ) ||
          next.reference_presets[0];
        return nextPreset?.id || "";
      });
    } catch (error) {
      setCatalogError(
        error instanceof Error ? error.message : "無法取得模型清單",
      );
    } finally {
      setCatalogLoading(false);
    }
  };

  useEffect(() => {
    void loadCatalog();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 只在整個 App 卸載時回收 blob URL。不可與輪詢的 effect 合併 —— 那個
  // effect 依賴 activeView，切換分頁時會連帶把播放中的音訊 URL 撤銷掉。
  useEffect(() => {
    const audioUrls = audioUrlsRef.current;
    return () => {
      streamingPlayerRef.current?.stop();
      audioUrls.forEach((audioUrl) => URL.revokeObjectURL(audioUrl));
      audioUrls.clear();
    };
  }, []);

  // 比較分頁不顯示歷史紀錄，輪詢在那裡是純粹的浪費（每次都會掃整個
  // 紀錄目錄）。
  useEffect(() => {
    if (activeView !== "studio") {
      return;
    }
    void loadGlobalHistory();
    const refreshTimer = window.setInterval(
      () => void loadGlobalHistory(),
      15_000,
    );
    const refreshOnFocus = () => void loadGlobalHistory();
    window.addEventListener("focus", refreshOnFocus);
    return () => {
      window.clearInterval(refreshTimer);
      window.removeEventListener("focus", refreshOnFocus);
    };
  }, [loadGlobalHistory, activeView]);

  useEffect(() => {
    if (!catalogError) {
      return;
    }
    const retryTimer = window.setTimeout(() => void loadCatalog(), 5000);
    return () => window.clearTimeout(retryTimer);
  }, [catalogError]);

  const chooseEngine = (engine: Engine) => {
    setEngineId(engine.id);
    const firstModel =
      engine.models.find((model) => model.online !== false) || engine.models[0];
    setModelId(firstModel?.id || "");
    setSpeakerId(firstModel?.speakers?.[0]?.id || "");
    setGenerateError("");
  };

  const chooseModel = (nextId: string) => {
    setModelId(nextId);
    const model = selectedEngine?.models.find((item) => item.id === nextId);
    setSpeakerId(model?.speakers?.[0]?.id || "");
  };

  const clearReference = () => {
    setReferenceAudio(undefined);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    const targetText = text.trim();
    if (!selectedEngine || !selectedModel || !targetText) {
      setGenerateError("請選擇模型並輸入合成文字");
      return;
    }
    if (selectedModel.online === false) {
      setGenerateError("選取的模型目前離線");
      return;
    }

    setGenerating(true);
    setGenerateError("");
    streamingPlayerRef.current?.stop();
    let streamPlayer: StreamingAudioPlayer | null = null;
    try {
      streamPlayer = streamingActive ? new StreamingAudioPlayer() : null;
      streamingPlayerRef.current = streamPlayer;
      await streamPlayer?.prepare();
      const request = {
        engineId: selectedEngine.id,
        modelId: selectedModel.id,
        text: targetText,
        controlInstruction: promptCloningActive ? "" : controlInstruction,
        promptText,
        referencePresetId,
        speakerId,
        cfgValue,
        inferenceTimesteps: steps,
        speed: effectiveSpeed,
        normalize,
        denoise,
        seed: !streamingActive && seed ? Number(seed) : undefined,
        referenceAudio,
      };
      const result = streamingActive
        ? await synthesizeStream(request, (chunk) =>
            streamPlayer?.enqueue(chunk),
          )
        : await synthesize(request);
      streamPlayer?.finish();
      const audioUrl = URL.createObjectURL(result.blob);
      audioUrlsRef.current.add(audioUrl);
      const item: HistoryItem = {
        id: result.historyId || createHistoryId(),
        text: targetText,
        engineId: result.engineId,
        engineLabel: selectedEngine.label,
        modelId: result.modelId,
        modelLabel: selectedModel.label,
        referenceLabel: formatReferenceLabel(
          referenceAudio,
          selectedReferencePreset,
        ),
        speakerLabel: selectedSpeaker?.name,
        seed: result.seed,
        cfgValue,
        inferenceTimesteps: steps,
        speed: effectiveSpeed,
        normalize,
        denoise,
        promptText: promptText.trim() || undefined,
        controlInstruction: controlInstruction.trim() || undefined,
        createdAt: new Date(),
        audioUrl,
        durationLabel: result.durationLabel,
      };
      setCurrentHistoryId(item.id);
      setAutoPlayHistoryId(streamingActive ? "" : item.id);
      setHistory((previous) => {
        const next = [item, ...previous];
        next.slice(HISTORY_LIMIT).forEach((old) => {
          URL.revokeObjectURL(old.audioUrl);
          audioUrlsRef.current.delete(old.audioUrl);
        });
        return next.slice(0, HISTORY_LIMIT);
      });
    } catch (error) {
      streamPlayer?.stop();
      setGenerateError(
        error instanceof Error ? error.message : "語音生成失敗",
      );
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="VoxCPM 360 首頁">
          <span className="brand-mark">
            <AudioLines size={22} strokeWidth={2.2} />
          </span>
          <span>
            <strong>VoxCPM 360</strong>
            <small>Speech Studio</small>
          </span>
        </a>
        <div className="topbar-actions">
          <span className="system-pill">
            <span
              className={`status-dot ${catalogError ? "offline" : "online"}`}
            />
            {catalogError ? "Gateway offline" : "System ready"}
          </span>
          <a className="legacy-link" href="/legacy/" target="_blank">
            舊版 Gradio
          </a>
        </div>
      </header>

      <main className="workspace">
        <section className="intro">
          <div>
            <span className="eyebrow">
              <Sparkles size={14} />
              MULTI-MODEL SPEECH LAB
            </span>
            <h1>
              {activeView === "studio"
                ? "讓每一種語言，找到自己的聲音。"
                : "用同一把尺，看懂每一次訓練。"}
            </h1>
            {activeView === "studio" ? (
              <p>
                在原生 VoxCPM2 與 Barbet 換腦模型之間自由切換，
                同時管理基礎模型與每一次訓練成果。
              </p>
            ) : (
              <p>
                依驗證集分組檢視訓練結果，在可比較的範圍內排序模型表現、收斂位置與容量。
              </p>
            )}
          </div>
          <div className="intro-actions">
            <nav className="view-switcher" aria-label="主要功能">
              <button
                type="button"
                className={activeView === "studio" ? "active" : ""}
                onClick={() => setActiveView("studio")}
                aria-current={activeView === "studio" ? "page" : undefined}
              >
                <Mic2 size={16} />
                語音生成
              </button>
              <button
                type="button"
                className={activeView === "comparison" ? "active" : ""}
                onClick={() => setActiveView("comparison")}
                aria-current={activeView === "comparison" ? "page" : undefined}
              >
                <BarChart3 size={16} />
                模型比較
              </button>
            </nav>
            {activeView === "studio" ? (
              <button
                className="icon-button refresh-button"
                onClick={() => void loadCatalog()}
                disabled={catalogLoading}
                title="重新整理模型"
              >
                <RefreshCw size={17} className={catalogLoading ? "spin" : ""} />
                重新掃描
              </button>
            ) : (
              <button
                className="icon-button refresh-button"
                onClick={() => setComparisonReloadKey((current) => current + 1)}
                title="重新載入模型比較資料"
              >
                <RefreshCw size={17} />
                重新載入
              </button>
            )}
          </div>
        </section>

        {activeView === "studio" && (
          <>
            {catalogError && (
              <div className="alert error-alert">
                <CircleAlert size={18} />
                <span>{catalogError}</span>
              </div>
            )}

            <form className="studio-grid" onSubmit={handleSubmit}>
          <aside className="control-panel panel">
            <div className="panel-heading">
              <span className="step-number">01</span>
              <div>
                <h2>選擇模型</h2>
                <p>先選文字語意大腦，再選訓練版本</p>
              </div>
            </div>

            <div className="engine-list">
              {catalogLoading && catalog.engines.length === 0 ? (
                <>
                  <div className="engine-skeleton" />
                  <div className="engine-skeleton" />
                </>
              ) : (
                catalog.engines.map((engine) => (
                  <button
                    type="button"
                    key={engine.id}
                    className={`engine-card ${
                      engine.id === engineId ? "selected" : ""
                    }`}
                    onClick={() => chooseEngine(engine)}
                  >
                    <span className="engine-icon">
                      {engine.family === "barbet" ? (
                        <BrainCircuit size={21} />
                      ) : (
                        <Cpu size={21} />
                      )}
                    </span>
                    <span className="engine-copy">
                      <span className="engine-title-row">
                        <strong>{engine.label}</strong>
                        <span
                          className={`mini-status ${
                            engine.online ? "online" : "offline"
                          }`}
                        >
                          {engine.online ? "ONLINE" : "OFFLINE"}
                        </span>
                      </span>
                      <small>{engine.description}</small>
                    </span>
                    {engine.id === engineId && (
                      <span className="selected-check">
                        <Check size={14} />
                      </span>
                    )}
                  </button>
                ))
              )}
            </div>

            <label className="field-label" htmlFor="model-version">
              模型版本
            </label>
            <div className="select-wrap">
              <select
                id="model-version"
                value={modelId}
                onChange={(event) => chooseModel(event.target.value)}
                disabled={!selectedEngine}
              >
                {selectedEngine?.models.map((model) => (
                  <option
                    value={model.id}
                    key={model.id}
                    disabled={model.online === false}
                  >
                    {model.label}
                    {model.loaded ? "（目前載入）" : ""}
                    {model.online === false ? "（離線）" : ""}
                  </option>
                ))}
              </select>
              <ChevronDown size={17} />
            </div>

            {selectedModel && (
              <div className="model-meta">
                <div>
                  <Server size={15} />
                  <span>
                    {selectedModel.kind.toUpperCase()}
                    {selectedModel.loaded ? " · LOADED" : ""}
                  </span>
                </div>
                {selectedModel.gpu && (
                  <div>
                    <Gauge size={15} />
                    <span>{selectedModel.gpu}</span>
                  </div>
                )}
                <p>{selectedModel.description}</p>
              </div>
            )}

            {selectedEngine?.capabilities.speaker_selection &&
              speakers.length > 0 && (
                <>
                  <label className="field-label" htmlFor="speaker">
                    指定語者
                  </label>
                  <div className="select-wrap">
                    <select
                      id="speaker"
                      value={speakerId}
                      onChange={(event) => setSpeakerId(event.target.value)}
                    >
                      <option value="">模型預設音色</option>
                      {speakers.map((speaker) => (
                        <option key={speaker.id} value={speaker.id}>
                          {speaker.name}
                        </option>
                      ))}
                    </select>
                    <ChevronDown size={17} />
                  </div>
                </>
              )}

            <div className="capability-note">
              <BrainCircuit size={17} />
              <p>
                {selectedEngine?.family === "barbet"
                  ? "新版本放進 models/barbet/<版本>，再按上方「重新掃描」。"
                  : "新 LoRA 或 FULL 放進 models/native/<版本>，再按上方「重新掃描」。"}
              </p>
            </div>
          </aside>

          <section className="composer panel">
            <div className="panel-heading">
              <span className="step-number">02</span>
              <div>
                <h2>編寫與生成</h2>
                <p>輸入文字，搭配授權的參考聲音</p>
              </div>
            </div>

            <div className="text-field">
              <textarea
                value={text}
                onChange={(event) => setText(event.target.value)}
                maxLength={2000}
                placeholder="輸入要合成的台語、華語或中英混合文字…"
                aria-label="合成文字"
              />
              <span className="character-count">{text.length} / 2000</span>
            </div>

            {(supportsControlInstruction || supportsPromptTranscript) && (
              <div
                className={`form-row ${
                  supportsControlInstruction && supportsPromptTranscript
                    ? ""
                    : "single-field"
                }`}
              >
                {supportsControlInstruction && (
                  <label>
                    <span className="field-label">聲音控制指令</span>
                    <input
                      value={controlInstruction}
                      onChange={(event) =>
                        setControlInstruction(event.target.value)
                      }
                      disabled={promptCloningActive}
                      aria-describedby={
                        promptCloningActive ? "control-instruction-help" : undefined
                      }
                      placeholder="例如：溫暖、沉穩、語速稍慢"
                    />
                    {promptCloningActive && (
                      <small
                        id="control-instruction-help"
                        className="field-helper"
                      >
                        使用參考音與逐字稿時由聲音克隆控制語氣；為避免模型朗讀指令，
                        此欄不會送入模型。
                      </small>
                    )}
                  </label>
                )}
                {supportsPromptTranscript && (
                  <label>
                    <span className="field-label">參考音訊逐字稿</span>
                    <input
                      value={promptText}
                      onChange={(event) => setPromptText(event.target.value)}
                      readOnly={!referenceAudio}
                      aria-readonly={!referenceAudio}
                      placeholder={
                        referenceAudio
                          ? "請填入上傳音檔的逐字稿，留空會失去聲音克隆效果"
                          : "選用內建聲音時自動帶入，不需手動填寫"
                      }
                    />
                  </label>
                )}
              </div>
            )}

            {selectedEngine?.capabilities.reference_audio && (
              <div className="upload-section">
                <label className="reference-preset-field">
                  <span className="field-label">內建參考聲音</span>
                  <div
                    className="select-wrap reference-preset-select"
                    data-state={referencePresetState}
                  >
                    <select
                      value={referencePresetId}
                      onChange={(event) =>
                        setReferencePresetId(event.target.value)
                      }
                      disabled={
                        catalogLoading ||
                        Boolean(referenceAudio) ||
                        catalog.reference_presets.length === 0
                      }
                      aria-busy={catalogLoading}
                      aria-invalid={
                        !catalogLoading && catalog.reference_presets.length === 0
                      }
                      aria-describedby="reference-preset-help"
                    >
                      {catalog.reference_presets.length === 0 && (
                        <option value="">沒有可用的內建聲音</option>
                      )}
                      {catalog.reference_presets.map((preset) => (
                        <option key={preset.id} value={preset.id}>
                          {`${preset.label}（${REFERENCE_AUDIO_LANGUAGE_LABELS[preset.language]}）`}
                        </option>
                      ))}
                    </select>
                    <ChevronDown size={17} />
                  </div>
                  <small
                    id="reference-preset-help"
                    className="reference-helper"
                    data-state={referenceAudio ? "custom" : "preset"}
                  >
                    {referenceAudio
                      ? "已使用自訂上傳音檔；請在下方「參考音訊逐字稿」填入其內容，留空會失去聲音克隆效果"
                      : selectedReferencePreset?.prompt_text
                        ? `逐字稿（已自動帶入）：${selectedReferencePreset.prompt_text}`
                        : selectedReferencePreset?.description ||
                          "未上傳音檔時會使用此內建聲音"}
                  </small>
                </label>
                <div className="reference-upload-field">
                  <span className="upload-alternative-label">
                    或上傳自訂參考音訊
                  </span>
                  {!referenceAudio ? (
                    <button
                      type="button"
                      className="upload-zone"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <span className="upload-icon">
                        <Upload size={21} />
                      </span>
                      <span>
                        <strong>上傳或拖入參考音檔</strong>
                        <small>WAV、MP3、M4A，建議 3 秒以上乾淨語音</small>
                      </span>
                    </button>
                  ) : (
                    <div className="file-chip">
                      <FileAudio size={20} />
                      <span>
                        <strong>{referenceAudio.name}</strong>
                        <small>
                          {(referenceAudio.size / 1024 / 1024).toFixed(2)} MB
                        </small>
                      </span>
                      <button
                        type="button"
                        onClick={clearReference}
                        title="移除參考音檔"
                      >
                        <X size={17} />
                      </button>
                    </div>
                  )}
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="audio/*"
                    hidden
                    onChange={(event) =>
                      setReferenceAudio(event.target.files?.[0])
                    }
                  />
                </div>
              </div>
            )}

            {selectedEngine?.capabilities.seed && (
              <div className="form-row single-field">
                <label>
                  <span className="field-label">
                    隨機種子{streamingActive ? "（串流停用）" : ""}
                  </span>
                  <span className="seed-control">
                    <input
                      type="number"
                      min="0"
                      max="2147483647"
                      step="1"
                      value={seed}
                      onChange={(event) => setSeed(event.target.value)}
                      placeholder={
                        streamingActive ? "串流模式不支援" : "留空則隨機"
                      }
                      disabled={streamingActive}
                    />
                    <button
                      type="button"
                      className="seed-reroll"
                      onClick={() => setSeed(createRandomSeed())}
                      disabled={streamingActive}
                      title="產生新的隨機種子"
                    >
                      <Dices size={16} />
                      重骰
                    </button>
                  </span>
                  <small className="field-helper" aria-live="polite">
                    {seedHelpText(streamingActive, seed)}
                  </small>
                </label>
              </div>
            )}

            <button
              type="button"
              className="advanced-toggle"
              onClick={() => setAdvancedOpen((open) => !open)}
            >
              <span>
                <SlidersHorizontal size={17} />
                進階參數
              </span>
              <ChevronDown
                size={17}
                className={advancedOpen ? "chevron-open" : ""}
              />
            </button>

            {advancedOpen && (
              <div className="advanced-panel">
                <label className="range-field">
                  <span>
                    <span className="field-label">CFG 引導強度</span>
                    <output>{cfgValue.toFixed(1)}</output>
                  </span>
                  <input
                    type="range"
                    min="1"
                    max="5"
                    step="0.1"
                    value={cfgValue}
                    onChange={(event) => setCfgValue(Number(event.target.value))}
                  />
                </label>
                <label className="range-field">
                  <span>
                    <span className="field-label">DiT 取樣步數</span>
                    <output>{steps}</output>
                  </span>
                  <input
                    type="range"
                    min="1"
                    max="30"
                    step="1"
                    value={steps}
                    onChange={(event) => setSteps(Number(event.target.value))}
                  />
                </label>
                <label className="range-field">
                  <span>
                    <span className="field-label">
                      語速（{streamingActive ? "串流固定" : "後製變速"}）
                    </span>
                    <output>{effectiveSpeed.toFixed(2)}x</output>
                  </span>
                  <input
                    type="range"
                    min="0.5"
                    max="2"
                    step="0.05"
                    value={effectiveSpeed}
                    onChange={(event) => setSpeed(Number(event.target.value))}
                    disabled={streamingActive}
                  />
                </label>
                <label className="toggle-row">
                  <input
                    type="checkbox"
                    checked={normalize}
                    onChange={(event) => setNormalize(event.target.checked)}
                  />
                  <span className="toggle-switch" />
                  <span>
                    <strong>文字正規化</strong>
                    <small>展開數字、日期與常見縮寫</small>
                  </span>
                </label>
                <label className="toggle-row">
                  <input
                    type="checkbox"
                    checked={denoise}
                    onChange={(event) => setDenoise(event.target.checked)}
                    disabled={!referenceAudio}
                  />
                  <span className="toggle-switch" />
                  <span>
                    <strong>參考音訊去噪</strong>
                    <small>上傳參考聲音後可使用</small>
                  </span>
                </label>
              </div>
            )}

            <label className="toggle-row streaming-toggle">
              <input
                type="checkbox"
                role="switch"
                checked={streamingActive}
                onChange={(event) => setStreaming(event.target.checked)}
                disabled={!supportsStreaming || generating}
                aria-describedby="streaming-help"
              />
              <span className="toggle-switch" />
              <span>
                <strong>串流生成</strong>
                <small id="streaming-help">
                  {supportsStreaming
                    ? "邊生成邊播放；語速固定 1.00x，隨機種子停用"
                    : "目前選取的引擎不支援串流"}
                </small>
              </span>
            </label>

            {generateError && (
              <div className="alert error-alert compact">
                <CircleAlert size={17} />
                <span>{generateError}</span>
              </div>
            )}

            <button
              className="generate-button"
              type="submit"
              disabled={
                generating ||
                !text.trim() ||
                !selectedModel ||
                selectedModel.online === false
              }
            >
              {generating ? (
                <>
                  <LoaderCircle size={19} className="spin" />
                  {streamingActive ? "正在串流播放…" : "正在生成語音…"}
                </>
              ) : (
                <>
                  <Mic2 size={19} />
                  生成語音
                </>
              )}
            </button>
          </section>

          <aside className="result-panel panel">
            <div className="panel-heading compact-heading">
              <span className="step-number">03</span>
              <div>
                <h2>輸出結果</h2>
                <p>即時預聽與下載</p>
              </div>
            </div>

            {currentHistoryItem ? (
              <div className="audio-result">
                <div className="result-art">
                  <div className="pulse-ring">
                    <AudioLines size={34} />
                  </div>
                  <span className="success-label">
                    <Check size={13} />
                    GENERATION COMPLETE
                  </span>
                </div>
                <div className="waveform-bars" aria-hidden="true">
                  {Array.from({ length: 38 }).map((_, index) => (
                    <span
                      key={index}
                      style={{
                        height: `${18 + ((index * 17) % 45)}%`,
                      }}
                    />
                  ))}
                </div>
                <audio
                  key={currentHistoryItem.id}
                  controls
                  src={currentHistoryItem.audioUrl}
                  autoPlay={autoPlayHistoryId === currentHistoryItem.id}
                />
                <dl className="result-metadata">
                  <div>
                    <dt>模型</dt>
                    <dd>
                      {currentHistoryItem.engineLabel}
                      <small>{currentHistoryItem.modelLabel}</small>
                    </dd>
                  </div>
                  <div>
                    <dt>參考音</dt>
                    <dd>{currentHistoryItem.referenceLabel}</dd>
                  </div>
                  {currentHistoryItem.speakerLabel && (
                    <div>
                      <dt>語者</dt>
                      <dd>{currentHistoryItem.speakerLabel}</dd>
                    </div>
                  )}
                  {currentHistoryItem.seed !== undefined && (
                    <div>
                      <dt>種子</dt>
                      <dd className="numeric-value">{currentHistoryItem.seed}</dd>
                    </div>
                  )}
                  <div>
                    <dt>生成參數</dt>
                    <dd>
                      CFG {currentHistoryItem.cfgValue.toFixed(1)} · {currentHistoryItem.inferenceTimesteps} steps
                      {currentHistoryItem.speed !== undefined &&
                      currentHistoryItem.speed !== 1
                        ? ` · ${currentHistoryItem.speed.toFixed(2)}x`
                        : ""}
                      {currentHistoryItem.normalize ? " · 正規化" : ""}
                      {currentHistoryItem.denoise ? " · 去噪" : ""}
                    </dd>
                  </div>
                  {currentHistoryItem.promptText && (
                    <div>
                      <dt>參考逐字稿</dt>
                      <dd>{currentHistoryItem.promptText}</dd>
                    </div>
                  )}
                  {currentHistoryItem.controlInstruction && (
                    <div>
                      <dt>聲音控制</dt>
                      <dd>{currentHistoryItem.controlInstruction}</dd>
                    </div>
                  )}
                  {currentHistoryItem.durationLabel && (
                    <div>
                      <dt>耗時</dt>
                      <dd className="numeric-value">{currentHistoryItem.durationLabel}</dd>
                    </div>
                  )}
                </dl>
                <a
                  className="download-button"
                  href={currentHistoryItem.audioUrl}
                  download={`voxcpm360-${currentHistoryItem.id}.wav`}
                >
                  <Download size={17} />
                  下載 WAV
                </a>
              </div>
            ) : (
              <div className="empty-result">
                <span>
                  <AudioLines size={30} />
                </span>
                <h3>等待第一段聲音</h3>
                <p>選擇模型、輸入文字後，生成結果會出現在這裡。</p>
              </div>
            )}

            <div className="session-history">
              <div className="history-title">
                <span>
                  <History size={16} />
                  全域生成紀錄
                </span>
                <small>{history.length} / {HISTORY_LIMIT}</small>
              </div>
              {history.length === 0 ? (
                <p className="history-empty">尚無生成紀錄</p>
              ) : (
                <div className="history-list">
                  {history.map((item) => (
                    <button
                      type="button"
                      key={item.id}
                      className={`history-item ${item.id === currentHistoryId ? "active" : ""}`}
                      onClick={() => {
                        setCurrentHistoryId(item.id);
                        setAutoPlayHistoryId(item.id);
                      }}
                    >
                      <span className="history-play">
                        <AudioLines size={16} />
                      </span>
                      <span className="history-copy">
                        <strong>{item.text}</strong>
                        <small>
                          {item.engineLabel} · {item.modelLabel}
                        </small>
                        <small>
                          參考：{item.referenceLabel}
                          {item.seed !== undefined ? ` · Seed ${item.seed}` : ""}
                        </small>
                      </span>
                      <span className="history-time">
                        <Clock3 size={12} />
                        {item.createdAt.toLocaleString("zh-TW", {
                          month: "2-digit",
                          day: "2-digit",
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </aside>
            </form>
          </>
        )}

        {activeView === "comparison" && (
          <ModelComparison reloadToken={comparisonReloadKey} />
        )}
      </main>
      <footer>
        <span>VoxCPM 360 · Multi-model inference gateway</span>
        <span>請只使用已取得授權的參考聲音</span>
      </footer>
    </div>
  );
}

export default App;
