import type {
  Catalog,
  HistoryItem,
  ModelRegistry,
  SynthesisRequest,
  SynthesisResult,
} from "./types";

interface GenerationHistoryRecord {
  id: string;
  text: string;
  engine_id: string;
  engine_label: string;
  model_id: string;
  model_label: string;
  reference_label: string;
  speaker_label?: string;
  seed?: number;
  cfg_value: number;
  inference_timesteps: number;
  speed?: number;
  normalize: boolean;
  denoise: boolean;
  prompt_text?: string;
  control_instruction?: string;
  duration_label?: string;
  created_at: string;
  audio_url: string;
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: string };
    return payload.detail || `請求失敗（${response.status}）`;
  } catch {
    return (await response.text()) || `請求失敗（${response.status}）`;
  }
}

export async function fetchCatalog(signal?: AbortSignal): Promise<Catalog> {
  const response = await fetch("/api/v1/catalog", { signal });
  if (!response.ok) {
    throw new Error(await errorMessage(response));
  }
  return response.json() as Promise<Catalog>;
}

export async function fetchModelRegistry(
  signal?: AbortSignal,
): Promise<ModelRegistry> {
  const response = await fetch("/api/v1/models/registry", { signal });
  if (!response.ok) {
    throw new Error(await errorMessage(response));
  }
  return response.json() as Promise<ModelRegistry>;
}

export async function fetchGenerationHistory(limit = 5): Promise<HistoryItem[]> {
  const response = await fetch(`/api/v1/history?limit=${limit}`);
  if (!response.ok) {
    throw new Error(await errorMessage(response));
  }
  const payload = (await response.json()) as { items: GenerationHistoryRecord[] };
  return payload.items.map((item) => ({
    id: item.id,
    text: item.text,
    engineId: item.engine_id,
    engineLabel: item.engine_label,
    modelId: item.model_id,
    modelLabel: item.model_label,
    referenceLabel: item.reference_label,
    speakerLabel: item.speaker_label,
    seed: item.seed,
    cfgValue: item.cfg_value,
    inferenceTimesteps: item.inference_timesteps,
    speed: item.speed,
    normalize: item.normalize,
    denoise: item.denoise,
    promptText: item.prompt_text,
    controlInstruction: item.control_instruction,
    durationLabel: item.duration_label,
    createdAt: new Date(item.created_at),
    audioUrl: item.audio_url,
  }));
}

export async function synthesize(
  request: SynthesisRequest,
): Promise<SynthesisResult> {
  const body = new FormData();
  body.set("engine_id", request.engineId);
  body.set("model_id", request.modelId);
  body.set("text", request.text);
  body.set("control_instruction", request.controlInstruction);
  body.set("prompt_text", request.promptText);
  body.set("reference_preset_id", request.referencePresetId);
  body.set("speaker_id", request.speakerId);
  body.set("cfg_value", String(request.cfgValue));
  body.set("inference_timesteps", String(request.inferenceTimesteps));
  body.set("speed", String(request.speed));
  body.set("normalize", String(request.normalize));
  body.set("denoise", String(request.denoise));
  if (request.seed !== undefined) {
    body.set("seed", String(request.seed));
  }
  if (request.referenceAudio) {
    body.set("reference_audio", request.referenceAudio);
  }

  const response = await fetch("/api/v1/synthesize", {
    method: "POST",
    body,
  });
  if (!response.ok) {
    throw new Error(await errorMessage(response));
  }

  return {
    blob: await response.blob(),
    historyId: response.headers.get("X-History-ID") || "",
    durationLabel: response.headers.get("X-Synthesis-Time") || undefined,
    engineId: response.headers.get("X-Model-Engine") || request.engineId,
    modelId: response.headers.get("X-Model-Version") || request.modelId,
    seed: response.headers.has("X-Random-Seed")
      ? Number(response.headers.get("X-Random-Seed"))
      : request.seed,
  };
}
