import torch

from scripts.train_voxcpm_finetune import encode_text, load_checkpoint, save_checkpoint


class _HfTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert text == "台語"
        assert add_special_tokens is False
        return [10, 20]


def test_encode_text_supports_hf_and_voxcpm_tokenizers():
    assert encode_text(_HfTokenizer(), "台語") == [10, 20]
    assert encode_text(lambda text: [len(text)], "台語") == [2]


def test_checkpoint_helpers_support_models_without_lora_config(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    save_checkpoint(model, optimizer, scheduler, tmp_path, step=7)

    assert (tmp_path / "step_0000007" / "model.safetensors").is_file()
    assert load_checkpoint(model, optimizer, scheduler, tmp_path) == 7
