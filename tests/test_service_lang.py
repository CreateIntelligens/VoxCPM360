"""台語／華語兩個服務共用程式，靠 VOXCPM_LANG 區分；nginx 依 ?lang= 分流。

這裡驗的是 gateway 這一側：送錯服務、lang 寫在 body 時要明確報錯，
而不是靜默念成另一種語言；回給呼叫端的網址要帶回 lang。
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api
import gateway.history
from test_api_gateway import FakeBarbetRuntime, FakeDemo, wait_for_batch

NGINX_TEMPLATE = Path(__file__).resolve().parents[1] / "nginx.template"


@pytest.fixture
def make_client(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")

    def _make(lang: str | None):
        if lang is None:
            monkeypatch.delenv("VOXCPM_LANG", raising=False)
        else:
            monkeypatch.setenv("VOXCPM_LANG", lang)
        demo = FakeDemo()
        app = api.create_app(demo, barbet_runtime=FakeBarbetRuntime())
        return TestClient(app), demo

    return _make


def test_default_service_is_taiwanese_and_labels_every_response(make_client):
    client, _ = make_client(None)
    with client:
        response = client.get("/api/v1/tts/health")
        tw_response = client.get("/api/v1/tts/health?lang=tw")

    assert response.status_code == 200
    assert response.headers["x-tts-lang"] == "tw"
    assert tw_response.status_code == 200


def test_query_lang_for_another_service_is_rejected(make_client):
    client, demo = make_client("tw")
    with client:
        response = client.post(
            "/api/v1/synthesize?lang=zh",
            data={"engine_id": "voxcpm2", "text": "測試"},
        )

    assert response.status_code == 400
    assert response.headers["x-tts-lang"] == "tw"
    assert "lang=zh" in response.json()["detail"]
    assert demo.calls == []


@pytest.mark.parametrize(
    ("path", "kwargs"),
    [
        ("/api/v1/synthesize", {"data": {"engine_id": "voxcpm2", "text": "測試", "lang": "zh"}}),
        ("/api/v1/synthesize/stream", {"data": {"engine_id": "voxcpm2", "text": "測試", "lang": "zh"}}),
        (
            "/api/v1/tts/synthesize",
            {"json": {"text": "測試", "voice_id": "voxcpm2-cosy-young-female-01", "lang": "zh"}},
        ),
        (
            "/api/v1/tts/synthesize/batch",
            {"json": {"items": [{"text": "測試", "voice_id": "voxcpm2-cosy-young-female-01"}], "lang": "zh"}},
        ),
    ],
)
def test_body_lang_that_does_not_match_the_service_is_rejected(make_client, path, kwargs):
    # 只寫在 body 的 lang 不會讓 nginx 換服務；台語服務收到就必須報錯。
    client, demo = make_client("tw")
    with client:
        response = client.post(path, **kwargs)

    assert response.status_code == 400
    assert "?lang=zh" in response.json()["detail"]
    assert demo.calls == []


def test_body_lang_matching_the_service_is_accepted(make_client):
    client, demo = make_client("zh")
    with client:
        response = client.post(
            "/api/v1/synthesize?lang=zh",
            data={"engine_id": "voxcpm2", "text": "測試", "lang": "zh"},
        )

    assert response.status_code == 200
    assert response.headers["x-tts-lang"] == "zh"
    assert len(demo.calls) == 1


def test_unknown_body_lang_is_rejected(make_client):
    client, _ = make_client("tw")
    with client:
        response = client.post(
            "/api/v1/synthesize",
            data={"engine_id": "voxcpm2", "text": "測試", "lang": "en"},
        )

    assert response.status_code == 400
    assert "tw 或 zh" in response.json()["detail"]


def test_mandarin_service_relabels_voices(make_client):
    client, _ = make_client("zh")
    with client:
        voices = client.get("/api/v1/tts/voices?lang=zh").json()["voices"]

    assert voices
    assert {voice["language"] for voice in voices} == {"zh-TW"}
    assert all("臺灣台語" not in voice["label"] for voice in voices)


def test_taiwanese_service_keeps_voice_language(make_client):
    client, _ = make_client("tw")
    with client:
        voices = client.get("/api/v1/tts/voices").json()["voices"]

    assert {voice["language"] for voice in voices} == {"nan-TW"}


def test_mandarin_batch_urls_carry_lang_back(make_client):
    # 輪詢與下載若沒帶 lang，nginx 會送去台語服務，那裡查不到這個 batch。
    client, _ = make_client("zh")
    with client:
        created = client.post(
            "/api/v1/tts/synthesize/batch?lang=zh",
            json={"items": [{"text": "測試", "voice_id": "voxcpm2-cosy-young-female-01"}]},
        )
        assert created.status_code == 202
        status_url = created.json()["status_url"]
        assert status_url.endswith("?lang=zh")
        batch = wait_for_batch(client, created.json()["batch_id"])

    audio_urls = [item["audio_url"] for item in batch["items"] if "audio_url" in item]
    assert audio_urls
    assert all(url.endswith("?lang=zh") for url in audio_urls)


def test_mandarin_history_audio_url_carries_lang(make_client):
    client, _ = make_client("zh")
    with client:
        synth = client.post(
            "/api/v1/synthesize?lang=zh",
            data={"engine_id": "voxcpm2", "text": "測試"},
        )
        assert synth.status_code == 200
        history = client.get("/api/v1/history?lang=zh").json()

    records = history["items"] if isinstance(history, dict) else history
    assert records
    assert records[0]["audio_url"].endswith("/audio?lang=zh")


def test_taiwanese_urls_stay_unchanged(make_client):
    client, _ = make_client("tw")
    with client:
        created = client.post(
            "/api/v1/tts/synthesize/batch",
            json={"items": [{"text": "測試", "voice_id": "voxcpm2-cosy-young-female-01"}]},
        )

    assert "?" not in created.json()["status_url"]


def test_invalid_service_lang_fails_at_startup(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    monkeypatch.setenv("VOXCPM_LANG", "en")
    with pytest.raises(RuntimeError, match="VOXCPM_LANG"):
        api.create_app(FakeDemo(), barbet_runtime=FakeBarbetRuntime())


def test_nginx_routes_by_query_lang_only_to_known_services():
    # nginx 的分流表與 gateway 認得的語言要一致，漏一邊就會靜默送錯服務。
    template = NGINX_TEMPLATE.read_text(encoding="utf-8")
    assert 'map $arg_lang $voxcpm_upstream' in template
    for line in ('""      app:8000;', "tw      app:8000;", "zh      app-zh:8000;", 'default "";'):
        assert line in template
    assert "proxy_pass http://$voxcpm_upstream;" in template


def test_base_model_is_listed_only_when_enabled(make_client, monkeypatch):
    # 華語服務沒有自訓模型，不列官方原版前端就沒有模型可選；台語服務照舊不列。
    client, _ = make_client("tw")
    with client:
        tw_models = client.get("/api/v1/catalog").json()["engines"][0]["models"]
    assert all(model["kind"] != "base" for model in tw_models)

    monkeypatch.setenv("VOXCPM_LIST_BASE_MODEL", "true")
    client, _ = make_client("zh")
    with client:
        zh_models = client.get("/api/v1/catalog?lang=zh").json()["engines"][0]["models"]
    assert [model["id"] for model in zh_models] == ["base::__base__"]
    assert zh_models[0]["loaded"] is True


def test_disabled_default_reference_sends_no_preset_transcript(make_client, monkeypatch):
    # 華語服務關掉台語預設參考音；只剩逐字稿時引擎會回錯（沒有參考音不能有 prompt_text）。
    monkeypatch.setenv("VOXCPM_DEFAULT_REFERENCE", "")
    client, demo = make_client("zh")
    with client:
        response = client.post(
            "/api/v1/synthesize?lang=zh",
            data={"engine_id": "voxcpm2", "text": "今天天氣很好"},
        )

    assert response.status_code == 200
    call = demo.calls[0]
    assert not call.get("reference_wav_path_input")
    assert not call.get("prompt_text")


@pytest.mark.parametrize("query", ["lang=ZH", "lang=Zh", "lang=&lang=tw"])
def test_query_lang_is_read_like_nginx(make_client, query):
    # nginx 的 map 不分大小寫、取第一個 lang（含空值）；這裡必須做一樣的判斷。
    lang = "zh" if "zh" in query.lower() else "tw"
    client, _ = make_client(lang)
    with client:
        response = client.get(f"/api/v1/tts/health?{query}")

    assert response.status_code == 200


def test_batch_item_lang_that_does_not_match_is_rejected(make_client):
    client, demo = make_client("tw")
    with client:
        response = client.post(
            "/api/v1/tts/synthesize/batch",
            json={"items": [{"text": "測試", "voice_id": "voxcpm2-cosy-young-female-01", "lang": "zh"}]},
        )

    assert response.status_code == 400
    assert demo.calls == []
