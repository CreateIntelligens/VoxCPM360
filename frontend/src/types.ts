export interface Speaker {
  id: string;
  name: string;
  desc?: string;
  is_custom?: boolean;
}

export interface ModelVersion {
  id: string;
  label: string;
  kind: "base" | "full" | "lora" | "checkpoint";
  description?: string;
  online?: boolean;
  loaded?: boolean;
  gpu?: string | null;
  speakers?: Speaker[];
}

export interface EngineCapabilities {
  control_instruction: boolean;
  prompt_transcript: boolean;
  reference_audio: boolean;
  speaker_selection: boolean;
  seed: boolean;
}

export interface Engine {
  id: string;
  label: string;
  family: "minicpm" | "barbet" | string;
  description: string;
  online: boolean;
  capabilities: EngineCapabilities;
  models: ModelVersion[];
}

export interface ReferenceAudioPreset {
  id: string;
  label: string;
  description: string;
  /** 內建參考音的逐字稿。未上傳自訂音檔時後端會自動帶入，使用者不必自己填。 */
  prompt_text: string;
}

export interface Catalog {
  engines: Engine[];
  reference_presets: ReferenceAudioPreset[];
  default_reference_preset_id: string;
}

export interface SynthesisRequest {
  engineId: string;
  modelId: string;
  text: string;
  controlInstruction: string;
  promptText: string;
  referencePresetId: string;
  speakerId: string;
  cfgValue: number;
  inferenceTimesteps: number;
  speed: number;
  normalize: boolean;
  denoise: boolean;
  seed?: number;
  referenceAudio?: File;
}

export interface SynthesisResult {
  blob: Blob;
  historyId: string;
  durationLabel?: string;
  engineId: string;
  modelId: string;
  seed?: number;
}

export interface HistoryItem {
  id: string;
  text: string;
  engineId: string;
  engineLabel: string;
  modelId: string;
  modelLabel: string;
  referenceLabel: string;
  speakerLabel?: string;
  seed?: number;
  cfgValue: number;
  inferenceTimesteps: number;
  // 早於語速功能的紀錄沒有這個欄位，故為選填。
  speed?: number;
  normalize: boolean;
  denoise: boolean;
  promptText?: string;
  controlInstruction?: string;
  createdAt: Date;
  audioUrl: string;
  durationLabel?: string;
}

export type ModelTrainingMethod =
  | "full-finetune"
  | "lora"
  | "bluemagpie-tslm"
  | "bluemagpie-bridge";

export interface ModelRegistryEntry {
  name: string;
  method: ModelTrainingMethod;
  arch: string;
  train_data: string;
  val_set: string;
  val_loss: number;
  best_epoch: number;
  best_step: number;
  lr: string;
  effective_batch: number;
  size_gb: number;
  lora_r?: number;
  note?: string;
}

export interface ModelRegistry {
  _val_sets: Record<string, string>;
  models: ModelRegistryEntry[];
}
