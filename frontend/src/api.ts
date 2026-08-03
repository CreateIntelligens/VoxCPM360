import type { Catalog, SynthesisRequest, SynthesisResult } from "./types";

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
    durationLabel: response.headers.get("X-Synthesis-Time") || undefined,
    engineId: response.headers.get("X-Model-Engine") || request.engineId,
    modelId: response.headers.get("X-Model-Version") || request.modelId,
  };
}
