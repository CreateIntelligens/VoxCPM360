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

const WAV_HEADER_BYTES = 44;

export interface StreamingAudioChunk {
  samples: Float32Array<ArrayBuffer>;
  sampleRate: number;
}

async function errorMessage(response: Response): Promise<string> {
  // body stream 只能讀一次：先取原文，再嘗試解析。
  // 原本先 json() 後在 catch 裡 text()，會拋
  // "body stream already read" 把真正的錯誤訊息蓋掉。
  let raw: string;
  try {
    raw = await response.text();
  } catch {
    return `請求失敗（${response.status}）`;
  }
  if (!raw) {
    return `請求失敗（${response.status}）`;
  }
  try {
    const payload = JSON.parse(raw) as { detail?: string; message?: string };
    return payload.detail || payload.message || raw;
  } catch {
    return raw;
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

function synthesisBody(request: SynthesisRequest): FormData {
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
  return body;
}

function synthesisResult(
  response: Response,
  request: SynthesisRequest,
  blob: Blob,
): SynthesisResult {
  return {
    blob,
    historyId: response.headers.get("X-History-ID") || "",
    durationLabel: response.headers.get("X-Synthesis-Time") || undefined,
    engineId: response.headers.get("X-Model-Engine") || request.engineId,
    modelId: response.headers.get("X-Model-Version") || request.modelId,
    seed: response.headers.has("X-Random-Seed")
      ? Number(response.headers.get("X-Random-Seed"))
      : request.seed,
  };
}

function wavBlob(
  pcmChunks: Uint8Array<ArrayBuffer>[],
  sampleRate: number,
): Blob {
  const dataSize = pcmChunks.reduce(
    (total, chunk) => total + chunk.byteLength,
    0,
  );
  const header = new ArrayBuffer(WAV_HEADER_BYTES);
  const view = new DataView(header);

  function writeText(offset: number, value: string): void {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  }

  writeText(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeText(36, "data");
  view.setUint32(40, dataSize, true);

  return new Blob([header, ...pcmChunks], { type: "audio/wav" });
}

function isStreamingWavHeader(header: Uint8Array): boolean {
  return (
    header[0] === 0x52 &&
    header[1] === 0x49 &&
    header[2] === 0x46 &&
    header[3] === 0x46 &&
    header[8] === 0x57 &&
    header[9] === 0x41 &&
    header[10] === 0x56 &&
    header[11] === 0x45
  );
}

export async function synthesize(
  request: SynthesisRequest,
): Promise<SynthesisResult> {
  const response = await fetch("/api/v1/synthesize", {
    method: "POST",
    body: synthesisBody(request),
  });
  if (!response.ok) {
    throw new Error(await errorMessage(response));
  }

  return synthesisResult(response, request, await response.blob());
}

export async function synthesizeStream(
  request: SynthesisRequest,
  onAudioChunk: (chunk: StreamingAudioChunk) => void,
): Promise<SynthesisResult> {
  const response = await fetch("/api/v1/synthesize/stream", {
    method: "POST",
    body: synthesisBody(request),
  });
  if (!response.ok) {
    throw new Error(await errorMessage(response));
  }

  const sampleRate = Number(response.headers.get("X-Sample-Rate"));
  if (!Number.isInteger(sampleRate) || sampleRate <= 0) {
    throw new Error("串流回應缺少有效的取樣率");
  }

  const pcmChunks: Uint8Array<ArrayBuffer>[] = [];
  const wavHeader = new Uint8Array(WAV_HEADER_BYTES);
  let headerOffset = 0;
  let pendingByte: number | undefined;

  function consume(incoming: Uint8Array): void {
    let offset = 0;
    if (headerOffset < WAV_HEADER_BYTES) {
      const headerBytes = Math.min(
        WAV_HEADER_BYTES - headerOffset,
        incoming.byteLength,
      );
      wavHeader.set(incoming.subarray(0, headerBytes), headerOffset);
      headerOffset += headerBytes;
      offset = headerBytes;
      if (
        headerOffset === WAV_HEADER_BYTES &&
        !isStreamingWavHeader(wavHeader)
      ) {
        throw new Error("串流端點回傳了無效的 WAV 標頭");
      }
    }

    if (offset >= incoming.byteLength) {
      return;
    }

    const pcm = new Uint8Array(incoming.byteLength - offset);
    pcm.set(incoming.subarray(offset));
    pcmChunks.push(pcm);
    let playable = pcm;
    if (pendingByte !== undefined) {
      const joined = new Uint8Array(pcm.byteLength + 1);
      joined[0] = pendingByte;
      joined.set(pcm, 1);
      playable = joined;
      pendingByte = undefined;
    }
    if (playable.byteLength % 2 !== 0) {
      pendingByte = playable[playable.byteLength - 1];
      playable = playable.subarray(0, playable.byteLength - 1);
    }
    if (playable.byteLength === 0) {
      return;
    }

    const view = new DataView(
      playable.buffer,
      playable.byteOffset,
      playable.byteLength,
    );
    const samples = new Float32Array(playable.byteLength / 2);
    for (let index = 0; index < samples.length; index += 1) {
      samples[index] = view.getInt16(index * 2, true) / 32768;
    }
    onAudioChunk({ samples, sampleRate });
  }

  if (response.body) {
    const reader = response.body.getReader();
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      consume(value);
    }
  } else {
    consume(new Uint8Array(await response.arrayBuffer()));
  }

  if (headerOffset !== WAV_HEADER_BYTES || pendingByte !== undefined) {
    throw new Error("串流音訊未完整傳輸");
  }
  if (pcmChunks.length === 0) {
    throw new Error("串流端點未回傳音訊資料");
  }

  return synthesisResult(response, request, wavBlob(pcmChunks, sampleRate));
}
