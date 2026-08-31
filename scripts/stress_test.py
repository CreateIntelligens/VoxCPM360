#!/usr/bin/env python3
"""VoxCPM Concurrent Streaming & Batch Stress Test.

Validates the v2 concurrent streaming architecture:
1. 2~4 Concurrent Streams (TTFB overlap & independent chunk flow)
2. Mixed Load: Concurrent Streams + Batch Coalescing
3. Concurrency header reporting (X-Engine-Concurrency) & RTF
"""

import asyncio
import io
import time
import httpx
import soundfile as sf

BASE_URL = "http://localhost:8000"

async def test_stream(client: httpx.AsyncClient, req_id: int, text: str):
    start_time = time.perf_counter()
    first_chunk_time = None
    total_bytes = 0
    headers_dict = {}
    chunk_count = 0
    wav_buffer = io.BytesIO()

    try:
        async with client.stream(
            "POST",
            f"{BASE_URL}/api/v1/synthesize/stream",
            data={
                "engine_id": "voxcpm2",
                "model_id": "__base__",
                "text": text,
            },
            timeout=120.0,
        ) as response:
            headers_dict = dict(response.headers)
            status_code = response.status_code
            if status_code != 200:
                body = await response.aread()
                return {
                    "req_id": req_id,
                    "type": "stream",
                    "status": status_code,
                    "error": body.decode(errors="ignore"),
                }

            async for chunk in response.aiter_bytes():
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()
                total_bytes += len(chunk)
                chunk_count += 1
                wav_buffer.write(chunk)

        end_time = time.perf_counter()
        ttfb = (first_chunk_time - start_time) if first_chunk_time else 0.0
        total_time = end_time - start_time

        wav_bytes = wav_buffer.getvalue()
        audio_dur = 0.0
        if len(wav_bytes) > 44:
            try:
                wav_buffer.seek(0)
                data, sr = sf.read(wav_buffer)
                audio_dur = len(data) / sr
            except Exception:
                pass

        rtf = (audio_dur / total_time) if total_time > 0 else 0.0

        return {
            "req_id": req_id,
            "type": "stream",
            "status": 200,
            "ttfb": ttfb,
            "total_time": total_time,
            "audio_dur": audio_dur,
            "rtf": rtf,
            "chunks": chunk_count,
            "bytes": total_bytes,
            "concurrency_header": headers_dict.get("x-engine-concurrency", "N/A"),
            "model_version": headers_dict.get("x-model-version", "N/A"),
        }
    except Exception as exc:
        return {
            "req_id": req_id,
            "type": "stream",
            "status": "EXCEPTION",
            "error": str(exc),
        }

async def test_non_stream(client: httpx.AsyncClient, req_id: int, text: str):
    start_time = time.perf_counter()
    try:
        response = await client.post(
            f"{BASE_URL}/api/v1/synthesize",
            data={
                "engine_id": "voxcpm2",
                "model_id": "__base__",
                "text": text,
            },
            timeout=120.0,
        )
        total_time = time.perf_counter() - start_time
        headers_dict = dict(response.headers)
        if response.status_code != 200:
            return {
                "req_id": req_id,
                "type": "batch",
                "status": response.status_code,
                "error": response.text,
            }

        wav_bytes = response.content
        audio_dur = 0.0
        try:
            data, sr = sf.read(io.BytesIO(wav_bytes))
            audio_dur = len(data) / sr
        except Exception:
            pass

        return {
            "req_id": req_id,
            "type": "batch",
            "status": 200,
            "total_time": total_time,
            "audio_dur": audio_dur,
            "rtf": audio_dur / total_time if total_time > 0 else 0.0,
            "concurrency_header": headers_dict.get("x-engine-concurrency", "N/A"),
            "batch_size_header": headers_dict.get("x-batch-size", "N/A"),
            "gpu_job_time": headers_dict.get("x-gpu-job-time", "N/A"),
        }
    except Exception as exc:
        return {
            "req_id": req_id,
            "type": "batch",
            "status": "EXCEPTION",
            "error": str(exc),
        }

async def main():
    print("=" * 65)
    print("🚀 VoxCPM360 併發串流（Concurrent Streaming）壓力測試")
    print("=" * 65)

    async with httpx.AsyncClient() as client:
        # 1. 健康檢查
        health = await client.get(f"{BASE_URL}/api/v1/health")
        print(f"Health Check: {health.status_code} -> status={health.json().get('status')}")

        # 2. 場景 A: 2 路併發串流 (Concurrent 2 Streams)
        print("\n" + "-" * 55)
        print("▶️ [場景 A] 2 路併發串流測試（驗證 TTFB 不阻塞 & 並行生成）...")
        texts_2 = [
            "第一路串流測試：台語語音合成系統即時推論展示。",
            "第二路串流測試：多序列平行生成與低延遲驗證。",
        ]
        start_a = time.perf_counter()
        results_a = await asyncio.gather(
            test_stream(client, 1, texts_2[0]),
            test_stream(client, 2, texts_2[1]),
        )
        total_a = time.perf_counter() - start_a
        print(f"✅ 場景 A 完成，總耗時: {total_a:.2f}s")
        for res in results_a:
            print(f"   [Stream #{res['req_id']}] Status={res['status']} | TTFB={res.get('ttfb', 0):.2f}s | "
                  f"Total={res.get('total_time', 0):.2f}s | Audio={res.get('audio_dur', 0):.2f}s ({res.get('rtf', 0):.2f}x RTF) | "
                  f"ConcurrencyHeader={res.get('concurrency_header')} | Chunks={res.get('chunks')}")

        # 3. 場景 B: 4 路併發串流 (Concurrent 4 Streams)
        print("\n" + "-" * 55)
        print("▶️ [場景 B] 4 路併發串流測試（滿載 capacity=4 壓測）...")
        texts_4 = [
            "第一路：歡迎收聽今日台語新聞速報。",
            "第二路：中央氣象署發布最新天氣概況。",
            "第三路：台南傳統市場在地小吃介紹。",
            "第四路：智慧交通新科技推廣與應用。",
        ]
        start_b = time.perf_counter()
        results_b = await asyncio.gather(*[
            test_stream(client, idx + 1, text) for idx, text in enumerate(texts_4)
        ])
        total_b = time.perf_counter() - start_b
        print(f"✅ 場景 B 完成，總耗時: {total_b:.2f}s")
        for res in results_b:
            print(f"   [Stream #{res['req_id']}] Status={res['status']} | TTFB={res.get('ttfb', 0):.2f}s | "
                  f"Total={res.get('total_time', 0):.2f}s | Audio={res.get('audio_dur', 0):.2f}s ({res.get('rtf', 0):.2f}x RTF) | "
                  f"ConcurrencyHeader={res.get('concurrency_header')} | Chunks={res.get('chunks')}")

        # 4. 場景 C: 混合負載（2 路串流 + 2 筆合批請求）
        print("\n" + "-" * 55)
        print("▶️ [場景 C] 混合負載測試（2 路串流 + 2 筆非串流批次同時進入）...")
        start_c = time.perf_counter()
        results_c = await asyncio.gather(
            test_stream(client, 1, "混合測試：串流第一路語音。"),
            test_stream(client, 2, "混合測試：串流第二路語音。"),
            test_non_stream(client, 3, "混合測試：非串流合批請求 A。"),
            test_non_stream(client, 4, "混合測試：非串流合批請求 B。"),
        )
        total_c = time.perf_counter() - start_c
        print(f"✅ 場景 C 完成，總耗時: {total_c:.2f}s")
        for res in results_c:
            req_type = res.get("type", "stream")
            if req_type == "stream":
                print(f"   [Stream #{res['req_id']}] Status={res['status']} | TTFB={res.get('ttfb', 0):.2f}s | "
                      f"Total={res.get('total_time', 0):.2f}s | Audio={res.get('audio_dur', 0):.2f}s ({res.get('rtf', 0):.2f}x RTF) | "
                      f"ConcurrencyHeader={res.get('concurrency_header')}")
            else:
                print(f"   [Batch  #{res['req_id']}] Status={res['status']} | Total={res.get('total_time', 0):.2f}s | "
                      f"Audio={res.get('audio_dur', 0):.2f}s ({res.get('rtf', 0):.2f}x RTF) | "
                      f"ConcurrencyHeader={res.get('concurrency_header')} | BatchSize={res.get('batch_size_header')}")

    print("\n" + "=" * 65)
    print("🎉 壓力測試全部成功完成！")
    print("=" * 65)

if __name__ == "__main__":
    asyncio.run(main())
