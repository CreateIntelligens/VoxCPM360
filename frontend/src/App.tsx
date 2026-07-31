import {
  AudioLines,
  BrainCircuit,
  Check,
  ChevronDown,
  CircleAlert,
  Clock3,
  Cpu,
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
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { fetchCatalog, synthesize } from "./api";
import type {
  Catalog,
  Engine,
  HistoryItem,
  ModelVersion,
} from "./types";

const DEFAULT_TEXT = "逐家好，歡迎使用 VoxCPM 360 多模型語音工作室。";

function createHistoryId(): string {
  if (typeof globalThis.crypto?.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function App() {
  const [catalog, setCatalog] = useState<Catalog>({ engines: [] });
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [catalogError, setCatalogError] = useState("");
  const [engineId, setEngineId] = useState("");
  const [modelId, setModelId] = useState("");
  const [speakerId, setSpeakerId] = useState("");
  const [text, setText] = useState(DEFAULT_TEXT);
  const [controlInstruction, setControlInstruction] = useState("");
  const [promptText, setPromptText] = useState("");
  const [referenceAudio, setReferenceAudio] = useState<File>();
  const [cfgValue, setCfgValue] = useState(2);
  const [steps, setSteps] = useState(10);
  const [normalize, setNormalize] = useState(false);
  const [denoise, setDenoise] = useState(false);
  const [seed, setSeed] = useState("");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [generateError, setGenerateError] = useState("");
  const [currentAudioUrl, setCurrentAudioUrl] = useState("");
  const [history, setHistory] = useState<HistoryItem[]>([]);
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
        nextEngine?.models.find((model) => model.online !== false) ||
        nextEngine?.models[0];
      setModelId(nextModel?.id || "");
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
    return () => {
      history.forEach((item) => URL.revokeObjectURL(item.audioUrl));
    };
    // The cleanup captures initial history intentionally; individual URLs are
    // revoked when history entries roll off.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
    try {
      const result = await synthesize({
        engineId: selectedEngine.id,
        modelId: selectedModel.id,
        text: targetText,
        controlInstruction,
        promptText,
        speakerId,
        cfgValue,
        inferenceTimesteps: steps,
        normalize,
        denoise,
        seed: seed ? Number(seed) : undefined,
        referenceAudio,
      });
      const audioUrl = URL.createObjectURL(result.blob);
      setCurrentAudioUrl(audioUrl);
      const item: HistoryItem = {
        id: createHistoryId(),
        text: targetText,
        engineLabel: selectedEngine.label,
        modelLabel: selectedModel.label,
        createdAt: new Date(),
        audioUrl,
        durationLabel: result.durationLabel,
      };
      setHistory((previous) => {
        const next = [item, ...previous];
        next.slice(5).forEach((old) => URL.revokeObjectURL(old.audioUrl));
        return next.slice(0, 5);
      });
    } catch (error) {
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
            <h1>讓每一種語言，找到自己的聲音。</h1>
            <p>
              在原生 VoxCPM2 與 Barbet 換腦模型之間自由切換，
              同時管理基礎模型與每一次訓練成果。
            </p>
          </div>
          <button
            className="icon-button refresh-button"
            onClick={() => void loadCatalog()}
            disabled={catalogLoading}
            title="重新整理模型"
          >
            <RefreshCw size={17} className={catalogLoading ? "spin" : ""} />
            重新掃描
          </button>
        </section>

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

            {selectedEngine?.capabilities.control_instruction && (
              <div className="form-row">
                <label>
                  <span className="field-label">聲音控制指令</span>
                  <input
                    value={controlInstruction}
                    onChange={(event) =>
                      setControlInstruction(event.target.value)
                    }
                    placeholder="例如：溫暖、沉穩、語速稍慢"
                  />
                </label>
                <label>
                  <span className="field-label">參考音訊逐字稿</span>
                  <input
                    value={promptText}
                    onChange={(event) => setPromptText(event.target.value)}
                    placeholder="有逐字稿時可做精準聲音複製"
                  />
                </label>
              </div>
            )}

            {selectedEngine?.capabilities.reference_audio && (
              <div className="upload-section">
                <span className="field-label">參考聲音</span>
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
                {selectedEngine?.capabilities.seed && (
                  <label>
                    <span className="field-label">隨機種子</span>
                    <input
                      type="number"
                      value={seed}
                      onChange={(event) => setSeed(event.target.value)}
                      placeholder="留空則隨機"
                    />
                  </label>
                )}
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
                  正在生成語音…
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

            {currentAudioUrl ? (
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
                <audio controls src={currentAudioUrl} autoPlay />
                <a
                  className="download-button"
                  href={currentAudioUrl}
                  download={`voxcpm360-${Date.now()}.wav`}
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
                  本次工作階段
                </span>
                <small>{history.length} / 5</small>
              </div>
              {history.length === 0 ? (
                <p className="history-empty">尚無生成紀錄</p>
              ) : (
                <div className="history-list">
                  {history.map((item) => (
                    <button
                      type="button"
                      key={item.id}
                      className="history-item"
                      onClick={() => setCurrentAudioUrl(item.audioUrl)}
                    >
                      <span className="history-play">
                        <AudioLines size={16} />
                      </span>
                      <span className="history-copy">
                        <strong>{item.text}</strong>
                        <small>
                          {item.engineLabel} · {item.modelLabel}
                        </small>
                      </span>
                      <span className="history-time">
                        <Clock3 size={12} />
                        {item.createdAt.toLocaleTimeString("zh-TW", {
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
      </main>
      <footer>
        <span>VoxCPM 360 · Multi-model inference gateway</span>
        <span>請只使用已取得授權的參考聲音</span>
      </footer>
    </div>
  );
}

export default App;
