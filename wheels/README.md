# flash-attn wheels

本目錄存放建置 image 時優先使用的本機 `flash-attn` wheel。cu128 與 cu130
產物檔名相同但 ABI 不相容，必須依 CUDA variant 分目錄：

```text
wheels/
├── cu128/
│   └── flash_attn-2.8.3-cp310-cp310-linux_<arch>.whl
└── cu130/
    └── flash_attn-2.8.3-cp310-cp310-linux_<arch>.whl
```

Dockerfile 只會讀取 `wheels/${TORCH_CUDA_VARIANT}/`，本機沒有對應 wheel 時，
會依序嘗試同 variant 的 GitHub Release，最後才從原始碼編譯。Release tag 固定為：

- `flash-attn-wheels-cu128`
- `flash-attn-wheels-cu130`

舊的 `flash-attn-wheels` tag 僅保留既有 2.6.3 cu128 產物，不再供新版
Dockerfile 自動下載。

GB10 編譯 cu130 wheel：

```bash
docker buildx build --target flash-wheel \
  --build-arg CUDA_BASE_IMAGE=nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04 \
  --build-arg TORCH_CUDA_VARIANT=cu130 \
  --build-arg TORCH_ARCH_LIST=12.0 \
  -o type=local,dest=wheels-cu130 .
```

導出後，將 wheel 放到 `wheels/cu130/` 才會被完整 image build 使用。切換
variant 時不必刪除另一套 wheel；目錄隔離會避免誤裝。
