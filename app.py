import asyncio
import inspect
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from typing import Any, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gradio as gr
import numpy as np
from funasr import AutoModel
from nanovllm_voxcpm import VoxCPM
from nanovllm_voxcpm.models.voxcpm2.config import LoRAConfig

import voxcpm
from voxcpm.lora_registry import (
    BASE_MODEL_KEY,
    LoraRegistry,
    build_nano_lora_config,
)
from voxcpm.model.utils import resolve_runtime_device


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------- Inline i18n (en + zh-CN only) ----------

_USAGE_INSTRUCTIONS_EN = (
    "**VoxCPM2 — Three Modes of Speech Generation:**\n\n"
    "🎨 **Voice Design** — Create a brand-new voice  \n"
    "No reference audio required. Describe the desired voice characteristics "
    "(gender, age, tone, emotion, pace …) in **Control Instruction**, and VoxCPM2 "
    "will craft a unique voice from your description alone.\n\n"
    "🎛️ **Controllable Cloning** — Clone a voice with optional style guidance  \n"
    "Upload a reference audio clip, then use **Control Instruction** to steer "
    "emotion, speaking pace, and overall style while preserving the original timbre.\n\n"
    "🎙️ **Ultimate Cloning** — Reproduce every vocal nuance through audio continuation  \n"
    "Turn on **Ultimate Cloning Mode** and provide (or auto-transcribe) the reference audio's transcript. "
    "The model treats the reference clip as a spoken prefix and seamlessly **continues** from it, faithfully preserving every vocal detail."
    "Note: This mode will disable Control Instruction."
)

_EXAMPLES_FOOTER_EN = (
    "---\n"
    "**💡 Voice Description Examples:**  \n"
    "Try the following Control Instructions to explore different voices:  \n\n"
    "**Example 1 — Gentle & Melancholic Girl**  \n"
    '`Control Instruction`: *"A young girl with a soft, sweet voice. '
    'Speaks slowly with a melancholic, slightly tsundere tone."*  \n'
    '`Target Text`: *"I never asked you to stay… It\'s not like I care or anything. '
    'But… why does it still hurt so much now that you\'re gone?"*  \n\n'
    "**Example 2 — Laid-Back Surfer Dude**  \n"
    '`Control Instruction`: *"Relaxed young male voice, slightly nasal, '
    'lazy drawl, very casual and chill."*  \n'
    '`Target Text`: *"Dude, did you see that set? The waves out there are totally gnarly today. '
    "Just catching barrels all morning — it's like, totally righteous, you know what I mean?\"*"
)

_USAGE_INSTRUCTIONS_ZH = (
    "**VoxCPM2 — 三种语音生成方式：**\n\n"
    "🎨 **声音设计（Voice Design）**  \n"
    "无需参考音频。在 **Control Instruction** 中描述目标音色特征"
    "（性别、年龄、语气、情绪、语速等），VoxCPM2 即可为你从零创造独一无二的声音。\n\n"
    "🎛️ **可控克隆（Controllable Cloning）**  \n"
    "上传参考音频，同时可选地使用 **Control Instruction** 来指定情绪、语速、风格等表达方式，"
    "在保留原始音色的基础上灵活控制说话风格。\n\n"
    "🎙️ **极致克隆（Ultimate Cloning）**  \n"
    "开启 **极致克隆模式** 并提供参考音频的文字内容（可自动识别）。"
    "模型会将参考音频视为已说出的前文，以**音频续写**的方式完整还原参考音频中的所有声音细节。"
    "注意：该模式与可控克隆模式互斥，将禁用Control Instruction。\n\n"
)

_EXAMPLES_FOOTER_ZH = (
    "---\n"
    "**💡 声音描述示例（中英文均可）：**  \n\n"
    "**示例 1 — 深宫太后**  \n"
    '`Control Instruction`: *"中老年女性，声音低沉阴冷，语速缓慢而有力，'
    '字字深思熟虑，带有深不可测的城府与威慑感。"*  \n'
    '`Target Text`: *"哀家在这深宫待了四十年，什么风浪没见过？你以为瞒得过哀家？"*  \n\n'
    "**示例 2 — 暴躁驾校教练**  \n"
    '`Control Instruction`: *"暴躁的中年男声，语速快，充满无奈和愤怒"*  \n'
    '`Target Text`: *"踩离合！踩刹车啊！你往哪儿开呢？前面是树你看不见吗？'
    '我教了你八百遍了，打死方向盘！你是不是想把车给我开到沟里去？"*  \n\n'
    "---\n"
    "**🗣️ 方言生成指南：**  \n"
    "要生成地道的方言语音，请在 **Target Text** 中直接使用方言词汇和句式，"
    "并在 **Control Instruction** 中描述方言特征。  \n\n"
    "**示例 — 广东话**  \n"
    '`Control Instruction`: *"粤语，中年男性，语气平淡"*  \n'
    '✅ 正确（粤语表达）：*"伙計，唔該一個A餐，凍奶茶少甜！"*  \n'
    '❌ 错误（普通话原文）：*"伙计，麻烦来一个A餐，冻奶茶少甜！"*  \n\n'
    "**示例 — 河南话**  \n"
    '`Control Instruction`: *"河南话，接地气的大叔"*  \n'
    '✅ 正确（河南话表达）：*"恁这是弄啥嘞？晌午吃啥饭？"*  \n'
    '❌ 错误（普通话原文）：*"你这是在干什么呢？中午吃什么饭？"*  \n\n'
    "🤖 **小技巧：** 不知道方言怎么写？可以用豆包、DeepSeek、Kimi 等 AI 助手"
    "将普通话翻译为方言文本，再粘贴到 Target Text 中即可。  \n\n"
)

_I18N_TRANSLATIONS = {
    "en": {
        "reference_audio_label": "🎤 Reference Audio (optional — upload for cloning)",
        "show_prompt_text_label": "🎙️ Ultimate Cloning Mode (transcript-guided cloning)",
        "show_prompt_text_info": "Auto-transcribes reference audio for every vocal nuance reproduced. Control Instruction will be disabled when active.",
        "prompt_text_label": "Transcript of Reference Audio (auto-filled via ASR, editable)",
        "prompt_text_placeholder": "The transcript of your reference audio will appear here …",
        "control_label": "🎛️ Control Instruction (optional — supports Chinese & English)",
        "control_placeholder": "e.g. A warm young woman / 年轻女性，温柔甜美 / Excited and fast-paced",
        "target_text_label": "✍️ Target Text — the content to speak",
        "generate_btn": "🔊 Generate Speech",
        "generated_audio_label": "Generated Audio",
        "advanced_settings_title": "⚙️ Advanced Settings",
        "ref_denoise_label": "Reference audio enhancement",
        "ref_denoise_info": "Apply ZipEnhancer denoising to the reference audio before cloning",
        "normalize_label": "Text normalization",
        "normalize_info": "Normalize numbers, dates, and abbreviations via wetext",
        "cfg_label": "CFG (guidance scale)",
        "cfg_info": "Higher → closer to the prompt / reference; lower → more creative variation",
        "dit_steps_label": "LocDiT flow-matching steps",
        "dit_steps_info": "LocDiT flow-matching steps — more steps → maybe better audio quality, but slower",
        "model_selector_label": "Inference model",
        "model_selector_info": "Choose the base model or the latest checkpoint from a LoRA training run.",
        "refresh_models_btn": "Rescan models",
        "usage_instructions": _USAGE_INSTRUCTIONS_EN,
        "examples_footer": _EXAMPLES_FOOTER_EN,
    },
    "zh-CN": {
        "reference_audio_label": "🎤 参考音频（可选 — 上传后用于克隆）",
        "show_prompt_text_label": "🎙️ 极致克隆模式（基于文本引导的极致克隆）",
        "show_prompt_text_info": "自动识别参考音频文本，完整还原音色、节奏、情感等全部声音细节。开启后 Control Instruction 将暂时禁用",
        "prompt_text_label": "参考音频内容文本（ASR 自动填充，可手动编辑）",
        "prompt_text_placeholder": "参考音频的文字内容将自动识别并显示在此处 …",
        "control_label": "🎛️ Control Instruction（可选 — 支持中英文描述）",
        "control_placeholder": "如：年轻女性，温柔甜美 / A warm young woman / 暴躁老哥，语速飞快",
        "target_text_label": "✍️ Target Text — 要合成的目标文本",
        "generate_btn": "🔊 开始生成",
        "generated_audio_label": "生成结果",
        "advanced_settings_title": "⚙️ 高级设置",
        "ref_denoise_label": "参考音频降噪增强",
        "ref_denoise_info": "克隆前使用 ZipEnhancer 对参考音频进行降噪处理",
        "normalize_label": "文本规范化",
        "normalize_info": "自动规范化数字、日期及缩写（基于 wetext）",
        "cfg_label": "CFG（引导强度）",
        "cfg_info": "数值越高 → 越贴合提示/参考音色；数值越低 → 生成风格更自由",
        "dit_steps_label": "LocDiT 流匹配迭代步数",
        "dit_steps_info": "LocDiT 流匹配生成迭代步数 — 步数越多 → 可能生成更好的音频质量，但速度变慢",
        "model_selector_label": "推論模型",
        "model_selector_info": "選擇基礎模型，或某次 LoRA 訓練的最新 checkpoint。",
        "refresh_models_btn": "重新掃描",
        "usage_instructions": _USAGE_INSTRUCTIONS_ZH,
        "examples_footer": _EXAMPLES_FOOTER_ZH,
    },
    "zh-Hans": None,  # alias, filled below
    "zh": None,       # alias, filled below
}
_I18N_TRANSLATIONS["zh-Hans"] = _I18N_TRANSLATIONS["zh-CN"]
_I18N_TRANSLATIONS["zh"] = _I18N_TRANSLATIONS["zh-CN"]

for _d in _I18N_TRANSLATIONS.values():
    if _d is not None:
        for _k, _v in _I18N_TRANSLATIONS["en"].items():
            _d.setdefault(_k, _v)

I18N = gr.I18n(**_I18N_TRANSLATIONS)

DEFAULT_TARGET_TEXT = (
    "VoxCPM2 is a creative multilingual TTS model from ModelBest, "
    "designed to generate highly realistic speech."
)

_CUSTOM_CSS = """
.logo-container {
    text-align: center;
    margin: 0.5rem 0 1rem 0;
}
.logo-container img {
    height: 80px;
    width: auto;
    max-width: 200px;
    display: inline-block;
}

/* Toggle switch style */
.switch-toggle {
    padding: 8px 12px;
    border-radius: 8px;
    background: var(--block-background-fill);
}
.switch-toggle input[type="checkbox"] {
    appearance: none;
    -webkit-appearance: none;
    width: 44px;
    height: 24px;
    background: #ccc;
    border-radius: 12px;
    position: relative;
    cursor: pointer;
    transition: background 0.3s ease;
    flex-shrink: 0;
}
.switch-toggle input[type="checkbox"]::after {
    content: "";
    position: absolute;
    top: 2px;
    left: 2px;
    width: 20px;
    height: 20px;
    background: white;
    border-radius: 50%;
    transition: transform 0.3s ease;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.switch-toggle input[type="checkbox"]:checked {
    background: var(--color-accent);
}
.switch-toggle input[type="checkbox"]:checked::after {
    transform: translateX(20px);
}
"""

_APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
)


# ---------- Model ----------

_OWNED_ENGINE_LOOP: Any = None
_OWNED_ENGINE_THREAD: Optional[threading.Thread] = None
_OWNED_ENGINE_LOCK = threading.Lock()


def _shutdown_owned_engine_loop(timeout: float = 10.0) -> None:
    """停掉專屬 loop 並**等它真正結束**。

    模型切換時 server.stop() 內部會 run_until_complete；只呼叫
    loop.stop() 而不等待，會與其形成競態
    （RuntimeError: Cannot run the event loop while another loop is
    running），讓卸載半途失敗、引擎留在半死狀態。
    """
    global _OWNED_ENGINE_LOOP, _OWNED_ENGINE_THREAD
    with _OWNED_ENGINE_LOCK:
        loop, thread = _OWNED_ENGINE_LOOP, _OWNED_ENGINE_THREAD
        _OWNED_ENGINE_LOOP = _OWNED_ENGINE_THREAD = None
    if loop is None:
        return
    if loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
    if thread is not None and thread.is_alive():
        thread.join(timeout=timeout)
    try:
        if not loop.is_running():
            loop.close()
    except Exception:  # 關閉失敗不該擋住模型切換
        logger.warning("owned engine loop close failed", exc_info=True)


def _ensure_owned_engine_loop() -> Any:
    """為「沒有自帶 loop 的裸 async pool」提供行程級的專屬 loop 執行緒。

    full checkpoint 切換路徑會拿到 AsyncVoxCPM2ServerPool 本身
    （既是 pool 也沒有 .loop）。沒有 loop 就只能走同步 fallback，
    而它的 generate 是 async generator，同步迭代必炸。
    模組級是因為呼叫點含 @staticmethod，取不到 self。
    """
    global _OWNED_ENGINE_LOOP
    with _OWNED_ENGINE_LOCK:
        loop = _OWNED_ENGINE_LOOP
        if loop is not None and loop.is_running():
            return loop
        global _OWNED_ENGINE_THREAD
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=loop.run_forever, name="voxcpm-owned-loop", daemon=True
        )
        thread.start()
        _OWNED_ENGINE_LOOP, _OWNED_ENGINE_THREAD = loop, thread
        return loop


class VoxCPMDemo:
    def __init__(self, model_id: str = "openbmb/VoxCPM2", device: str = "auto") -> None:
        self.device = resolve_runtime_device(device, "cuda")
        self.optimize = os.environ.get("VOXCPM_OPTIMIZE", "false").lower() == "true"
        self.gpu_memory_utilization = float(
            os.environ.get("VOXCPM_GPU_MEMORY_UTILIZATION", "0.35")
        )
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("VOXCPM_GPU_MEMORY_UTILIZATION must be in (0, 1]")
        self.max_generate_length = int(
            os.environ.get("VOXCPM_MAX_GENERATE_LENGTH", "2000")
        )
        if self.max_generate_length < 1:
            raise ValueError("VOXCPM_MAX_GENERATE_LENGTH must be >= 1")
        self.max_audio_text_ratio = float(
            os.environ.get("VOXCPM_MAX_AUDIO_TEXT_RATIO", "6.0")
        )
        if self.max_audio_text_ratio <= 0:
            raise ValueError("VOXCPM_MAX_AUDIO_TEXT_RATIO must be > 0")
        logger.info(
            "Running VoxCPM on device: %s (optimize=%s, gpu_memory_utilization=%.2f, "
            "max_generate_length=%d, max_audio_text_ratio=%.1f)",
            self.device,
            self.optimize,
            self.gpu_memory_utilization,
            self.max_generate_length,
            self.max_audio_text_ratio,
        )

        self.asr_model_id = "iic/SenseVoiceSmall"
        self.asr_device = "cuda:0" if self.device.startswith("cuda") else "cpu"
        self.asr_model: Optional[AutoModel] = None

        roots_setting = os.environ.get(
            "VOXCPM_LORA_ROOTS",
            "/app/models/native:/app/checkpoints",
        )
        legacy_root = os.environ.get("VOXCPM_LORA_ROOT")
        lora_roots = [
            Path(value)
            for value in (legacy_root or roots_setting).split(os.pathsep)
            if value.strip()
        ]
        if not lora_roots:
            lora_roots = [Path("/app/models/native"), Path("/app/checkpoints")]
        self.lora_registry = LoraRegistry(
            lora_roots[0],
            additional_roots=lora_roots[1:],
        )
        logger.info("Discovered %d LoRA checkpoints.", len(self.lora_registry.checkpoints))

        self.voxcpm_server = None
        self._server_loop_thread: Optional[threading.Thread] = None
        self._server_loop_lock = threading.Lock()
        self._model_id = model_id
        self.denoiser = None
        self.text_normalizer = None
        self.zipenhancer_model_path = "iic/speech_zipenhancer_ans_multiloss_16k_base"

    def _call_engine_sync(self, server: Any, method_name: str, *args: Any) -> Any:
        """同步呼叫引擎方法，與專屬 loop 執行緒相容。

        loop 執行緒啟動後，nano-vLLM 的同步包裝（內部 run_until_complete）
        會撞上運轉中的 loop 直接拋錯——此處改以 run_coroutine_threadsafe
        提交對應的 async pool 方法。loop 未運轉（舊 runtime）則走原同步路徑。
        """
        server_loop = getattr(server, "loop", None)
        pool = getattr(server, "server_pool", None)
        # 裸 async pool（full checkpoint 路徑）：自己就是 pool、沒有 .loop，
        # 其方法是 coroutine function，同步呼叫會拿到未 await 的 coroutine。
        if pool is None and inspect.iscoroutinefunction(
            getattr(server, method_name, None)
        ):
            pool = server
            server_loop = server_loop or _ensure_owned_engine_loop()
        if (
            server_loop is not None
            and server_loop.is_running()
            and pool is not None
        ):
            future = asyncio.run_coroutine_threadsafe(
                getattr(pool, method_name)(*args), server_loop
            )
            return future.result(timeout=300)
        return getattr(server, method_name)(*args)

    def _ensure_server_loop_running(self) -> None:
        server = getattr(self, "voxcpm_server", None)
        if server is None:
            try:
                server = self.get_or_load_voxcpm()
            except Exception:
                return
        if server is None:
            return
        server_loop = getattr(server, "loop", None)
        if server_loop is None:
            return
        loop_lock = getattr(self, "_server_loop_lock", None)
        if loop_lock is None:
            self._server_loop_lock = threading.Lock()
            loop_lock = self._server_loop_lock
        with loop_lock:
            loop_thread = getattr(self, "_server_loop_thread", None)
            if not server_loop.is_running() and (
                loop_thread is None or not loop_thread.is_alive()
            ):
                self._server_loop_thread = threading.Thread(
                    target=server_loop.run_forever,
                    daemon=True,
                    name="voxcpm-loop",
                )
                self._server_loop_thread.start()

    def get_or_load_voxcpm(self):
        if self.voxcpm_server is not None:
            self._ensure_server_loop_running()
            return self.voxcpm_server
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            # 縱深防禦：呼叫端若在 event loop 上，from_pretrained 會偵測到
            # running loop 而回傳「裸 async pool」（無 .loop／.server_pool），
            # 同步橋接全滅。改到乾淨執行緒載入，保證 Sync 包裝。
            box: list[Any] = []

            def _load_off_loop() -> None:
                try:
                    box.append(self.get_or_load_voxcpm())
                except BaseException as exc:  # noqa: BLE001
                    box.append(exc)

            loader = threading.Thread(
                target=_load_off_loop, name="voxcpm-model-load"
            )
            loader.start()
            loader.join()
            outcome = box[0]
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        logger.info(f"Loading nano-vllm model: {self._model_id}")
        devices = [0]
        if "cuda" in self.device:
            if ":" in self.device:
                try:
                    devices = [int(self.device.split(":")[-1])]
                except ValueError:
                    devices = [0]

        enforce_eager = not self.optimize
        runtime_lora_config = build_nano_lora_config(
            self.lora_registry.checkpoints
        )
        # Reserve enough adapter capacity for checkpoints copied in after the
        # server has started. This keeps ordinary rank/projection variations
        # hot-loadable without restarting the base model.
        runtime_lora_config["max_lora_rank"] = max(
            runtime_lora_config["max_lora_rank"],
            int(os.environ.get("VOXCPM_MAX_LORA_RANK", "128")),
        )
        runtime_lora_config["enable_proj"] = True
        lora_config = LoRAConfig(**runtime_lora_config)
        self.voxcpm_server = VoxCPM.from_pretrained(
            self._model_id,
            # nano-vLLM 只在建構時接受 diffusion 步數；per-request 的
            # inference_timesteps 到不了引擎（generate 簽名沒有它），
            # 所以真正的控制點是這個部署層環境變數。
            inference_timesteps=int(
                os.environ.get("VOXCPM_INFERENCE_TIMESTEPS", "10")
            ),
            max_num_batched_tokens=8192,
            max_num_seqs=16,
            max_model_len=4096,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=enforce_eager,
            devices=devices,
            lora_config=lora_config,
        )
        logger.info("nano-vllm model loaded successfully.")
        self._ensure_server_loop_running()
        return self.voxcpm_server

    def stop_voxcpm(self) -> None:
        """Stop nano-vLLM worker processes and release their GPU allocation."""
        server = getattr(self, "voxcpm_server", None)
        self.voxcpm_server = None
        lora_registry = getattr(self, "lora_registry", None)
        if lora_registry is not None:
            lora_registry.reset_registrations()
        if server is None:
            return
        server_loop = getattr(server, "loop", None)
        if server_loop is not None and server_loop.is_running():
            server_loop.call_soon_threadsafe(server_loop.stop)
            with self._server_loop_lock:
                loop_thread = self._server_loop_thread
                if loop_thread is not None and loop_thread.is_alive():
                    loop_thread.join(timeout=5.0)
                self._server_loop_thread = None
        model_id = getattr(self, "_model_id", "unknown")
        logger.info("Stopping nano-vllm model: %s", model_id)
        # server.stop() 內部走 run_until_complete；若我們的專屬 loop 仍在
        # 執行緒中運轉，會拋 "Cannot run the event loop while another loop
        # is running"，讓模型切換半途失敗（GB10 full checkpoint 實測）。
        _shutdown_owned_engine_loop()
        try:
            stop = getattr(server, "stop", None)
            if callable(stop):
                stop()
        finally:
            import gc

            gc.collect()
        logger.info("nano-vllm model stopped.")

    def get_or_load_asr_model(self) -> AutoModel:
        if self.asr_model is not None:
            return self.asr_model
        logger.info(
            f"Loading ASR model: {self.asr_model_id} on device: {self.asr_device}"
        )
        self.asr_model = AutoModel(
            model=self.asr_model_id,
            disable_update=True,
            log_level="DEBUG",
            device=self.asr_device,
        )
        logger.info("ASR model loaded successfully.")
        return self.asr_model

    def prompt_wav_recognition(self, prompt_wav: Optional[str]) -> str:
        if prompt_wav is None:
            return ""
        res = self.get_or_load_asr_model().generate(
            input=prompt_wav,
            language="auto",
            use_itn=True,
        )
        return res[0]["text"].split("|>")[-1]

    def _prepare_tts_generation(
        self,
        server: Any,
        temp_files: list[str],
        latent_cache: dict[str, Any],
        *,
        text_input: str,
        control_instruction: str = "",
        reference_wav_path_input: Optional[str] = None,
        prompt_text: str = "",
        cfg_value_input: float = 2.0,
        do_normalize: bool = True,
        denoise: bool = True,
        inference_timesteps: int = 10,
        seed: Optional[int] = None,
        model_selection: str = BASE_MODEL_KEY,
    ) -> dict[str, Any]:
        text = (text_input or "").strip()
        if not text:
            raise ValueError("Please input text to synthesize.")

        lora_name = self.lora_registry.ensure_registered(server, model_selection)

        prompt_text_clean = (prompt_text or "").strip() or None
        control = (control_instruction or "").strip()
        control = re.sub(r"[()（）]", "", control).strip()
        # Voice cloning（音訊 + 對應逐字稿）與文字控制不可同時使用。
        # control 並非獨立條件，而是直接前綴到 target text；兩者並用時部分模型
        # 會把「用台語說／雀躍女生」真的念出來。API 端也會清空，這裡再做一層
        # 防護，涵蓋舊版 UI 或其他直接呼叫者。
        if reference_wav_path_input and prompt_text_clean:
            control = ""
        final_text = f"({control}){text}" if control else text

        if reference_wav_path_input and prompt_text_clean:
            logger.info(f"[Voice Cloning] prompt_wav + prompt_text + reference_wav")
        elif reference_wav_path_input:
            logger.info(f"[Voice Control] reference_wav only")
        else:
            logger.info(f"[Voice Design] control: {control[:50] if control else 'None'}...")

        # 1. Text Normalization
        if do_normalize:
            if self.text_normalizer is None:
                logger.info("Loading TextNormalizer...")
                from voxcpm.utils.text_normalize import TextNormalizer
                self.text_normalizer = TextNormalizer()
            final_text = self.text_normalizer.normalize(final_text)

        # 2. Denoising reference audio
        actual_ref_path = reference_wav_path_input or None
        if denoise and actual_ref_path:
            if self.denoiser is None:
                logger.info("Loading ZipEnhancer denoiser...")
                from voxcpm.zipenhancer import ZipEnhancer
                self.denoiser = ZipEnhancer(self.zipenhancer_model_path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                temp_files.append(tmp.name)
            logger.info(f"Denoising reference audio {actual_ref_path} -> {temp_files[-1]}")
            self.denoiser.enhance(actual_ref_path, output_path=temp_files[-1])
            actual_ref_path = temp_files[-1]

        # 3. Encode latents. Reuse them within one batch when every item uses
        # the same built-in reference clip.
        ref_audio_latents = None
        prompt_latents = None
        if actual_ref_path:
            encoded = latent_cache.get(actual_ref_path)
            if encoded is None:
                with open(actual_ref_path, "rb") as f:
                    wav_bytes = f.read()
                logger.info(f"Encoding latents for reference audio: {len(wav_bytes)} bytes")
                ext = os.path.splitext(actual_ref_path)[1].lstrip('.') or "wav"
                encoded = self._call_engine_sync(
                    server, "encode_latents", wav_bytes, ext
                )
                latent_cache[actual_ref_path] = encoded
            if prompt_text_clean:
                prompt_latents = encoded
                ref_audio_latents = encoded
            else:
                ref_audio_latents = encoded

        # 4. Prepare nano-vLLM request.
        text_length = max(1, len(final_text))
        length_from_text = int(
            text_length * getattr(self, "max_audio_text_ratio", 6.0) + 10
        )
        max_generate_length = min(
            getattr(self, "max_generate_length", 2000), length_from_text
        )
        logger.info(
            "Generating audio with vLLM engine for text: '%s...' "
            "(max_generate_length=%d)",
            final_text[:80],
            max_generate_length,
        )
        generate_kwargs = {
            "target_text": final_text,
            "prompt_latents": prompt_latents,
            "prompt_text": prompt_text_clean if prompt_text_clean else "",
            "cfg_value": float(cfg_value_input),
            "max_generate_length": max_generate_length,
            "temperature": 1.0,
            "lora_name": lora_name,
        }

        # Inspect signature to be safe & compatible with both VoxCPM versions.
        sig = inspect.signature(server.generate)
        if "ref_audio_latents" in sig.parameters:
            generate_kwargs["ref_audio_latents"] = ref_audio_latents
        if "inference_timesteps" in sig.parameters:
            generate_kwargs["inference_timesteps"] = int(inference_timesteps)
        # 引擎支援 per-request seed（z_noise 經 derive_step_seed 逐步派生）。
        if seed is not None and "seed" in sig.parameters:
            generate_kwargs["seed"] = int(seed)
        return generate_kwargs

    @staticmethod
    def _generate_tts_requests(
        server: Any,
        requests: list[dict[str, Any]],
        *,
        return_exceptions: bool = False,
    ) -> list[np.ndarray | Exception]:
        def concatenate_chunks(chunks: list[np.ndarray]) -> np.ndarray:
            if not chunks:
                raise RuntimeError(
                    "Failed to generate audio (empty stream from backend)"
                )
            return np.concatenate(chunks, axis=0)

        async_pool = getattr(server, "server_pool", None)
        server_loop = getattr(server, "loop", None)
        # 部分路徑（如 full checkpoint 切換）拿到的是裸的 async pool，
        # 本身就是 pool、且沒有自帶 loop —— 此時借用 demo 的專屬 loop
        # 執行緒。否則會落到同步 fallback，對 async generator 迭代而炸
        # （'async_generator' object is not iterable）。
        if async_pool is None and inspect.isasyncgenfunction(
            getattr(server, "generate", None)
        ):
            async_pool = server
            server_loop = server_loop or _ensure_owned_engine_loop()
        if async_pool is None or server_loop is None:
            # 落到同步 fallback 前先記錄實情：某些 runtime 版本的 server
            # 物件不帶 server_pool/loop，其 generate 卻是 async generator，
            # 同步迭代會拋 "'async_generator' object is not iterable"。
            logger.warning(
                "engine bridge fallback: server=%s pool=%s loop=%s",
                type(server).__name__,
                type(async_pool).__name__ if async_pool is not None else None,
                type(server_loop).__name__ if server_loop is not None else None,
            )
        # 單一請求也必須走 async pool：專屬 loop 執行緒啟動後，同步包裝
        # generate 的 run_until_complete 會撞上運轉中的 loop 直接拋錯。
        if async_pool is not None and server_loop is not None:

            async def collect(request: dict[str, Any]) -> np.ndarray:
                chunks = [chunk async for chunk in async_pool.generate(**request)]
                return concatenate_chunks(chunks)

            async def collect_all() -> list[np.ndarray | Exception]:
                results = await asyncio.gather(
                    *(collect(request) for request in requests),
                    return_exceptions=True,
                )
                if return_exceptions:
                    isolated_results: list[np.ndarray | Exception] = []
                    for result in results:
                        if isinstance(result, asyncio.CancelledError):
                            error = RuntimeError("Batched generation was cancelled")
                            error.__cause__ = result
                            isolated_results.append(error)
                        else:
                            isolated_results.append(result)
                    return isolated_results
                # Wait for every submitted backend request before surfacing an
                # error. Returning early would release the API's GPU gate while
                # another request from this batch could still be running.
                errors = [result for result in results if isinstance(result, BaseException)]
                if errors:
                    raise RuntimeError(f"{len(errors)} of {len(requests)} batched generations failed") from errors[0]
                return [result for result in results if isinstance(result, np.ndarray)]

            # 引擎互動一律透過 run_coroutine_threadsafe 提交至專屬 loop 執行緒。
            # 禁令：禁止在併發路徑使用 server_loop.run_until_complete。
            if server_loop.is_running():
                future = asyncio.run_coroutine_threadsafe(collect_all(), server_loop)
                return future.result()
            return server_loop.run_until_complete(collect_all())

        results = []
        for request in requests:
            try:
                chunks = list(server.generate(**request))
                results.append(concatenate_chunks(chunks))
            except Exception as exc:
                if not return_exceptions:
                    raise
                results.append(exc)
        return results

    def generate_tts_audio_batch(
        self,
        requests: list[dict[str, Any]],
        *,
        return_exceptions: bool = False,
    ) -> list[tuple[int, np.ndarray] | Exception]:
        if not requests:
            return []
        server = self.get_or_load_voxcpm()
        temp_files: list[str] = []
        latent_cache: dict[str, Any] = {}
        try:
            if not return_exceptions:
                prepared = [
                    self._prepare_tts_generation(
                        server,
                        temp_files,
                        latent_cache,
                        **request,
                    )
                    for request in requests
                ]
                audio_results = self._generate_tts_requests(server, prepared)
                sample_rate = int(self._call_engine_sync(server, "get_model_info")["sample_rate"])
                return [(sample_rate, audio) for audio in audio_results]

            prepared: list[dict[str, Any]] = []
            prepared_indices: list[int] = []
            indexed_results: dict[int, tuple[int, np.ndarray] | Exception] = {}
            for index, request in enumerate(requests):
                try:
                    prepared.append(
                        self._prepare_tts_generation(
                            server,
                            temp_files,
                            latent_cache,
                            **request,
                        )
                    )
                    prepared_indices.append(index)
                except Exception as exc:  # noqa: BLE001 - preserve per-request failure
                    indexed_results[index] = exc

            audio_results = self._generate_tts_requests(
                server,
                prepared,
                return_exceptions=True,
            )
            sample_rate = int(self._call_engine_sync(server, "get_model_info")["sample_rate"])
            for index, result in zip(
                prepared_indices,
                audio_results,
                strict=True,
            ):
                indexed_results[index] = (
                    result if isinstance(result, Exception) else (sample_rate, result)
                )
            return [indexed_results[index] for index in range(len(requests))]
        finally:
            for tmp_path in temp_files:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    def generate_tts_audio_stream(
        self, request: dict[str, Any]
    ) -> Iterator[np.ndarray]:
        """逐段產生單一 nano-vLLM TTS 請求，並綁定暫存檔生命週期。"""
        server = self.get_or_load_voxcpm()
        temp_files: list[str] = []
        latent_cache: dict[str, Any] = {}
        try:
            prepared = self._prepare_tts_generation(
                server,
                temp_files,
                latent_cache,
                **request,
            )
            yielded = False
            async_pool = getattr(server, "server_pool", None)
            server_loop = getattr(server, "loop", None)
            # 與批次路徑同理：裸 async pool 沒有自帶 loop，需借用專屬 loop
            # 執行緒，否則落到同步分支對 async generator 迭代而炸。
            if async_pool is None and inspect.isasyncgenfunction(
                getattr(server, "generate", None)
            ):
                async_pool = server
                server_loop = server_loop or _ensure_owned_engine_loop()
            if async_pool is not None and server_loop is not None:
                self._ensure_server_loop_running()
                async_generator = async_pool.generate(**prepared)
                try:
                    while True:
                        if server_loop.is_running():
                            future = asyncio.run_coroutine_threadsafe(
                                async_generator.__anext__(),
                                server_loop,
                            )
                            try:
                                chunk = future.result()
                            except StopAsyncIteration:
                                break
                        else:
                            try:
                                chunk = server_loop.run_until_complete(
                                    async_generator.__anext__()
                                )
                            except StopAsyncIteration:
                                break
                        yielded = True
                        yield np.asarray(chunk, dtype=np.float32)
                finally:
                    # sync wrapper 的 close 不保證會取消底層 async CUDA request。
                    # 等 aclose 完成後 worker 才能離開，GPU gate 才可安全釋放。
                    if server_loop.is_running():
                        future = asyncio.run_coroutine_threadsafe(
                            async_generator.aclose(),
                            server_loop,
                        )
                        try:
                            future.result(timeout=10.0)
                        except Exception:
                            pass
                    else:
                        server_loop.run_until_complete(async_generator.aclose())
            else:
                generator = server.generate(**prepared)
                try:
                    for chunk in generator:
                        yielded = True
                        yield np.asarray(chunk, dtype=np.float32)
                finally:
                    close = getattr(generator, "close", None)
                    if callable(close):
                        close()
            if not yielded:
                raise RuntimeError(
                    "Failed to generate audio (empty stream from backend)"
                )
        finally:
            for tmp_path in temp_files:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    def generate_tts_audio(
        self,
        text_input: str,
        control_instruction: str = "",
        reference_wav_path_input: Optional[str] = None,
        prompt_text: str = "",
        cfg_value_input: float = 2.0,
        do_normalize: bool = True,
        denoise: bool = True,
        inference_timesteps: int = 10,
        model_selection: str = BASE_MODEL_KEY,
        seed: Optional[int] = None,
    ) -> Tuple[int, np.ndarray]:
        return self.generate_tts_audio_batch(
            [
                {
                    "seed": seed,
                    "text_input": text_input,
                    "control_instruction": control_instruction,
                    "reference_wav_path_input": reference_wav_path_input,
                    "prompt_text": prompt_text,
                    "cfg_value_input": cfg_value_input,
                    "do_normalize": do_normalize,
                    "denoise": denoise,
                    "inference_timesteps": inference_timesteps,
                    "model_selection": model_selection,
                }
            ]
        )[0]

    def warmup_voxcpm(
        self,
        *,
        reference_path: Optional[str] = None,
        prompt_text: str = "",
    ) -> None:
        """Run the common inference paths once before accepting traffic."""
        started_at = time.perf_counter()
        logger.info("Starting VoxCPM2 inference warmup")

        self.generate_tts_audio(
            text_input="暖機。",
            do_normalize=False,
            denoise=False,
            seed=0,
        )
        logger.info(
            "VoxCPM2 no-reference warmup completed in %.2fs",
            time.perf_counter() - started_at,
        )

        if reference_path:
            reference_started_at = time.perf_counter()
            self.generate_tts_audio(
                text_input="暖機。",
                reference_wav_path_input=reference_path,
                prompt_text=prompt_text,
                do_normalize=False,
                denoise=False,
                seed=0,
            )
            logger.info(
                "VoxCPM2 reference warmup completed in %.2fs",
                time.perf_counter() - reference_started_at,
            )
        else:
            logger.warning(
                "Default reference audio is unavailable; skipping reference warmup"
            )

        logger.info(
            "VoxCPM2 inference warmup completed in %.2fs",
            time.perf_counter() - started_at,
        )


# ---------- UI ----------

def create_demo_interface(demo: VoxCPMDemo):
    gr.set_static_paths(paths=[Path.cwd().absolute() / "assets"])

    def _generate(
        text: str,
        control_instruction: str,
        ref_wav: Optional[str],
        use_prompt_text: bool,
        prompt_text_value: str,
        cfg_value: float,
        do_normalize: bool,
        denoise: bool,
        dit_steps: int,
        model_selection: str,
    ):
        actual_prompt_text = prompt_text_value.strip() if use_prompt_text else ""
        actual_control = "" if use_prompt_text else control_instruction
        sr, wav_np = demo.generate_tts_audio(
            text_input=text,
            control_instruction=actual_control,
            reference_wav_path_input=ref_wav,
            prompt_text=actual_prompt_text,
            cfg_value_input=cfg_value,
            do_normalize=do_normalize,
            denoise=denoise,
            inference_timesteps=int(dit_steps),
            model_selection=model_selection,
        )
        return (sr, wav_np)

    def _refresh_models(current_selection: str):
        demo.lora_registry.refresh()
        choices = demo.lora_registry.choices()
        valid_selections = {value for _, value in choices}
        if current_selection in valid_selections:
            selection = current_selection
        else:
            selection = BASE_MODEL_KEY
        return (
            gr.update(
                choices=choices,
                value=selection,
            ),
            demo.lora_registry.describe(selection),
        )

    def _on_toggle_instant(checked):
        """Instant UI toggle — no ASR, no blocking."""
        if checked:
            return (
                gr.update(visible=True, value="", placeholder="Recognizing reference audio..."),
                gr.update(visible=False),
            )
        return (
            gr.update(visible=False),
            gr.update(visible=True, interactive=True),
        )

    def _run_asr_if_needed(checked, audio_path):
        """Run ASR after the UI has updated. Only when toggled ON."""
        if not checked or not audio_path:
            return gr.update()
        try:
            logger.info("Running ASR on reference audio...")
            asr_text = demo.prompt_wav_recognition(audio_path)
            logger.info(f"ASR result: {asr_text[:60]}...")
            return gr.update(value=asr_text)
        except Exception as e:
            logger.warning(f"ASR recognition failed: {e}")
            return gr.update(value="")

    with gr.Blocks() as interface:
        gr.HTML(
            '<div class="logo-container">'
            '<img src="/gradio_api/file=assets/voxcpm_logo.png" alt="VoxCPM Logo">'
            "</div>"
        )

        gr.Markdown(I18N("usage_instructions"))

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    model_selector = gr.Dropdown(
                        choices=demo.lora_registry.choices(),
                        value=BASE_MODEL_KEY,
                        label=I18N("model_selector_label"),
                        info=I18N("model_selector_info"),
                        interactive=True,
                        scale=4,
                    )
                    refresh_models = gr.Button(
                        I18N("refresh_models_btn"),
                        variant="secondary",
                        scale=1,
                    )
                model_status = gr.Markdown(
                    demo.lora_registry.describe(BASE_MODEL_KEY)
                )
                reference_wav = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label=I18N("reference_audio_label"),
                )
                show_prompt_text = gr.Checkbox(
                    value=False,
                    label=I18N("show_prompt_text_label"),
                    info=I18N("show_prompt_text_info"),
                    elem_classes=["switch-toggle"],
                )
                prompt_text = gr.Textbox(
                    value="",
                    label=I18N("prompt_text_label"),
                    placeholder=I18N("prompt_text_placeholder"),
                    lines=2,
                    visible=False,
                )
                control_instruction = gr.Textbox(
                    value="",
                    label=I18N("control_label"),
                    placeholder=I18N("control_placeholder"),
                    lines=2,
                )
                text = gr.Textbox(
                    value=DEFAULT_TARGET_TEXT,
                    label=I18N("target_text_label"),
                    lines=3,
                )

                with gr.Accordion(I18N("advanced_settings_title"), open=False):
                    DoDenoisePromptAudio = gr.Checkbox(
                        value=False,
                        label=I18N("ref_denoise_label"),
                        elem_classes=["switch-toggle"],
                        info=I18N("ref_denoise_info"),
                    )
                    DoNormalizeText = gr.Checkbox(
                        value=False,
                        label=I18N("normalize_label"),
                        elem_classes=["switch-toggle"],
                        info=I18N("normalize_info"),
                    )
                    cfg_value = gr.Slider(
                        minimum=1.0,
                        maximum=3.0,
                        value=2.0,
                        step=0.1,
                        label=I18N("cfg_label"),
                        info=I18N("cfg_info"),
                    )
                    dit_steps = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1,
                        label=I18N("dit_steps_label"),
                        info=I18N("dit_steps_info"),
                    )

                run_btn = gr.Button(I18N("generate_btn"), variant="primary", size="lg")

            with gr.Column():
                audio_output = gr.Audio(label=I18N("generated_audio_label"))
                gr.Markdown(I18N("examples_footer"))

        model_selector.change(
            fn=demo.lora_registry.describe,
            inputs=[model_selector],
            outputs=[model_status],
            show_progress=False,
        )
        refresh_models.click(
            fn=_refresh_models,
            inputs=[model_selector],
            outputs=[model_selector, model_status],
            show_progress="minimal",
        )

        show_prompt_text.change(
            fn=_on_toggle_instant,
            inputs=[show_prompt_text],
            outputs=[prompt_text, control_instruction],
        ).then(
            fn=_run_asr_if_needed,
            inputs=[show_prompt_text, reference_wav],
            outputs=[prompt_text],
        )

        run_btn.click(
            fn=_generate,
            inputs=[
                text,
                control_instruction,
                reference_wav,
                show_prompt_text,
                prompt_text,
                cfg_value,
                DoNormalizeText,
                DoDenoisePromptAudio,
                dit_steps,
                model_selector,
            ],
            outputs=[audio_output],
            show_progress=True,
            api_name="generate",
        )

    return interface

def run_demo(
    server_name: str = "0.0.0.0",
    server_port: int = 8808,
    show_error: bool = True,
    model_id: str = "openbmb/VoxCPM2",
    device: str = "auto",
):
    demo = VoxCPMDemo(model_id=model_id, device=device)
    
    # Pre-preload and warm up models on startup to avoid delay on first user inference
    logger.info("Pre-warming/Preloading models to device...")
    try:
        demo.get_or_load_voxcpm()
        demo.get_or_load_asr_model()
        logger.info("Models preloaded successfully!")
    except Exception as e:
        logger.warning(f"Failed to preload models during startup: {e}")

    # Real warmup: run actual inference once so any JIT/torch.compile/nvrtc errors
    # surface at startup (not on a user's first request) and the first user
    # generation is fast. Covers both the no-reference path and the reference
    # path (encode_latents -> VAE encoder), which exercise different kernels.
    try:
        import time as _t
        print("[warmup] starting inference warmup (compiles torch.compile graphs)...", flush=True)
        _t0 = _t.time()
        # 1) Voice Design path: text only, no reference audio.
        demo.generate_tts_audio(
            text_input="Warmup.",
            do_normalize=False,
            denoise=False,
            inference_timesteps=4,
        )
        print(f"[warmup] no-reference path done ({_t.time()-_t0:.1f}s)", flush=True)
        # 2) Voice Control path: tiny dummy reference audio -> encode_latents/VAE.
        import tempfile, soundfile as sf
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            warm_ref = tmp.name
        try:
            sf.write(warm_ref, np.zeros(16000, dtype=np.float32), 16000)
            demo.generate_tts_audio(
                text_input="Warmup.",
                reference_wav_path_input=warm_ref,
                do_normalize=False,
                denoise=False,
                inference_timesteps=4,
            )
        finally:
            try:
                os.remove(warm_ref)
            except OSError:
                pass
        print(f"[warmup] complete; inference paths ready (total {_t.time()-_t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"[warmup] FAILED (server will still start): {e!r}", flush=True)
        
    interface = create_demo_interface(demo)
    interface.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name=server_name,
        server_port=server_port,
        show_error=show_error,
        i18n=I18N,
        theme=_APP_THEME,
        css=_CUSTOM_CSS,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id", type=str, default="openbmb/VoxCPM2",
        help="Local path or HuggingFace repo ID (default: openbmb/VoxCPM2)",
    )
    parser.add_argument("--port", type=int, default=8808, help="Server port")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Runtime device: auto, cpu, mps, cuda, or cuda:N (default: auto)",
    )
    args = parser.parse_args()
    run_demo(model_id=args.model_id, server_port=args.port, device=args.device)
