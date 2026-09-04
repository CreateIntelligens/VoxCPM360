# flash-attn wheels

建置 image 時，Dockerfile 依序尋找 flash-attn wheel：

1. 本目錄（平放，不分子目錄）
2. GitHub Release `flash-attn-wheels-${TORCH_CUDA_VARIANT}`
3. 都沒有則從原始碼編譯，約一小時

比對的是含版本與架構的**完整檔名**，例如
`flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl`。cu128 與 cu130 的產物檔名
規則相同但 ABI 不相容，靠 `FLASH_ATTN_VERSION` 區分（cu128→2.6.3、
cu130→2.8.3），因此本目錄放錯 variant 的 wheel 不會被誤裝。

平常不需要在本目錄放任何東西 —— 兩個架構的 wheel 都已上傳 Release，
build 會自動下載。

## 產生新的 wheel

換 CUDA 版本或新增 CPU 架構時才需要：

```bash
docker buildx build --target flash-wheel \
  --build-arg CUDA_BASE_IMAGE=nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04 \
  --build-arg TORCH_CUDA_VARIANT=cu130 \
  --build-arg FLASH_ATTN_VERSION=2.8.3 \
  --build-arg TORCH_ARCH_LIST="8.6;9.0;12.0" \
  --build-arg TORCH_VERSION=2.13.0 \
  -o type=local,dest=wheels-out .
```

`TORCH_VERSION` 決定 wheel 綁定的 ABI，務必與部署時安裝的 torch 一致 ——
兩者不符會在 `import flash_attn` 當場失敗。各棧的完整組合與存放位置：

| CUDA variant | flash-attn | torch | torchaudio | Release tag |
|---|---|---|---|---|
| cu130（現行） | 2.8.3 | 2.13.0 | 2.11.0 | [`flash-attn-wheels-cu130`](https://github.com/CreateIntelligens/VoxCPM360/releases/tag/flash-attn-wheels-cu130) |
| cu128 | 2.6.3 | 2.11.0 | 2.11.0 | [`flash-attn-wheels-cu128`](https://github.com/CreateIntelligens/VoxCPM360/releases/tag/flash-attn-wheels-cu128) |

torchaudio 的版號自 2.12 起與 torch 脫鉤，兩者不必同號。每個 tag 各有
`linux_x86_64` 與 `linux_aarch64` 兩顆。

手動取得（正常情況下 build 會自動下載，不需要這步）：

```bash
gh release download flash-attn-wheels-cu130 -R CreateIntelligens/VoxCPM360 \
  --pattern 'flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl' --dir wheels/
```

導出後上傳到對應的 Release，其他機器即可免編譯：

```bash
gh release upload flash-attn-wheels-cu130 -R CreateIntelligens/VoxCPM360 \
  wheels-out/flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl
```

> BuildKit 會回收舊的編譯層，`--target flash-wheel` 不保證命中快取。
> 導出前先確認是否真的需要重編。
