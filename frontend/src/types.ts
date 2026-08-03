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
  normalize: boolean;
  denoise: boolean;
  seed?: number;
  referenceAudio?: File;
}

export interface SynthesisResult {
  blob: Blob;
  durationLabel?: string;
  engineId: string;
  modelId: string;
}

export interface HistoryItem {
  id: string;
  text: string;
  engineLabel: string;
  modelLabel: string;
  createdAt: Date;
  audioUrl: string;
  durationLabel?: string;
}
