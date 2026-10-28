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
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from nanovllm_voxcpm import VoxCPM
from nanovllm_voxcpm.models.voxcpm2.config import LoRAConfig

from runtime_config import (
    read_default_inference_timesteps,
    require_per_request_timesteps,
)

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


DEFAULT_GPU_MEMORY_GB = 10.0
# 沒有 CUDA（單元測試、CPU 機器）時算不出比例；nano-vLLM 反正只跑 CUDA，
# 這個值只是佔位，不會真的拿去配記憶體。
_FALLBACK_GPU_MEMORY_UTILIZATION = 0.35


def _total_gpu_memory_gb(device: str) -> Optional[float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        index = int(device.split(":")[-1]) if ":" in device else 0
        return torch.cuda.get_device_properties(index).total_memory / 1024**3
    except Exception:  # noqa: BLE001
        return None


def resolve_gpu_memory_utilization(
    device: str, environ: Optional[dict[str, str]] = None
) -> float:
    """把「這台最多給 VoxCPM 幾 GB」換算成 nano-vLLM 要的比例。

    比例寫死在 compose 會跟著機器變：0.35 在 16 GB 卡是 5.6 GB、在 119 GB
    統一記憶體的 GB10 卻是 42 GB，每換一台都要重算。改成用 GB 設定，
    VOXCPM_GPU_MEMORY_UTILIZATION 仍可明確指定比例覆蓋。
    """
    env = os.environ if environ is None else environ
    explicit = (env.get("VOXCPM_GPU_MEMORY_UTILIZATION") or "").strip()
    if explicit:
        fraction = float(explicit)
        if not 0 < fraction <= 1:
            raise ValueError("VOXCPM_GPU_MEMORY_UTILIZATION must be in (0, 1]")
        return fraction
    budget_gb = float(env.get("VOXCPM_GPU_MEMORY_GB") or DEFAULT_GPU_MEMORY_GB)
    if budget_gb <= 0:
        raise ValueError("VOXCPM_GPU_MEMORY_GB must be > 0")
    total_gb = _total_gpu_memory_gb(device)
    if total_gb is None:
        return _FALLBACK_GPU_MEMORY_UTILIZATION
    return min(1.0, round(budget_gb / total_gb, 4))


class EngineWorkerDied(RuntimeError):
    """nano-vLLM 推論子程序已不在（例如被 kernel OOM killer 殺掉）。"""


def _engine_processes(server: Any) -> Optional[list[Any]]:
    """取出 server 底下所有推論子程序；拿不到（假物件、舊 runtime）回 None。"""
    pool = getattr(server, "server_pool", None) or server
    servers = getattr(pool, "servers", None)
    if not isinstance(servers, (list, tuple)):
        return None
    processes = [getattr(item, "process", None) for item in servers]
    if not processes or any(proc is None for proc in processes):
        return None
    return processes


def engine_alive(server: Any) -> Optional[bool]:
    """True = 子程序都在；False = 至少一個已退出；None = 無法判斷。

    worker 被 OOM killer 殺掉時父行程不會收到任何通知，送進 queue 的請求
    永遠等不到回應。沒有這個檢查，health 照回 ok、每個請求卡滿 300 秒。
    """
    processes = _engine_processes(server)
    if processes is None:
        return None
    for proc in processes:
        if getattr(proc, "exitcode", None) is not None:
            return False
        is_alive = getattr(proc, "is_alive", None)
        if callable(is_alive) and not is_alive():
            return False
    return True


class VoxCPMDemo:
    def __init__(self, model_id: str = "openbmb/VoxCPM2", device: str = "auto") -> None:
        read_default_inference_timesteps()
        require_per_request_timesteps()
        self.device = resolve_runtime_device(device, "cuda")
        self.optimize = os.environ.get("VOXCPM_OPTIMIZE", "false").lower() == "true"
        self.gpu_memory_utilization = resolve_gpu_memory_utilization(self.device)
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
        if engine_alive(server) is False:
            raise EngineWorkerDied(
                f"nano-vllm worker is not running (method={method_name})"
            )
        if (
            server_loop is not None
            and server_loop.is_running()
            and pool is not None
        ):
            future = asyncio.run_coroutine_threadsafe(
                getattr(pool, method_name)(*args), server_loop
            )
            # 等待期間子程序若死掉，queue 永遠不會回應；每秒看一次存活，
            # 死了就立刻失敗，不要等滿 300 秒。
            deadline = time.monotonic() + 300
            while True:
                try:
                    return future.result(timeout=1.0)
                except FutureTimeoutError:
                    if engine_alive(server) is False:
                        future.cancel()
                        raise EngineWorkerDied(
                            f"nano-vllm worker died while handling {method_name}"
                        ) from None
                    if time.monotonic() >= deadline:
                        raise
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

    def worker_alive(self) -> Optional[bool]:
        """已載入的推論子程序是否還活著；未載入或無法判斷回 None。"""
        server = getattr(self, "voxcpm_server", None)
        if server is None:
            return None
        return engine_alive(server)

    def get_or_load_voxcpm(self):
        if self.voxcpm_server is not None:
            if self.worker_alive() is False:
                # 子程序死了但父行程還在：清掉殘骸重載，讓下一個請求自己復原，
                # 而不是對著死 queue 等 300 秒。
                logger.error(
                    "nano-vllm worker for %s is gone; reloading", self._model_id
                )
                self.stop_voxcpm()
            else:
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
        # nano-vLLM 只接受 GPU 編號，沒有 device type 的概念，且硬依賴
        # flash-attn。放行 cpu/mps 只會讓服務宣稱跑在該裝置上卻仍佔用 GPU 0。
        if not self.device.startswith("cuda"):
            raise ValueError(
                f"device {self.device!r} is not supported: the nano-vLLM runtime "
                "requires CUDA. Set VOXCPM_DEVICE to 'auto', 'cuda' or 'cuda:N'."
            )
        logger.info(f"Loading nano-vllm model: {self._model_id}")
        devices = [0]
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
            # 此步數是預設值，也是 CUDA graph 的固定加速路徑。
            # 逐請求的其他步數由 runner 分組後以 eager 執行。
            inference_timesteps=read_default_inference_timesteps(),
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

            import torch

            # 卸載後歸還顯存，與 BarbetRuntime._release_model 對稱。模型切換
            # 本就要重載數 GB，多這一步不影響延遲，但能避免新舊模型的配置
            # 模式差異在 16GB 卡上累積成碎片。
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        logger.info("nano-vllm model stopped.")

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
        if "inference_timesteps" not in sig.parameters:
            raise RuntimeError(
                "nano-vLLM lacks per-request timesteps; rebuild the app image "
                "with scripts/patch_nanovllm_timesteps.py"
            )
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
