"""請求欄位寫進 log 前要跳脫換行，否則可以偽造出一整行假紀錄。"""

import logging

from fastapi.testclient import TestClient

import api
from gateway.log_safe import log_value
from test_api_gateway import FakeBarbetRuntime, FakeDemo


def test_log_value_escapes_line_breaks():
    assert log_value("a\r\nb\nc") == "a\\r\\nb\\nc"
    assert log_value(None) == "None"


def test_received_log_cannot_be_split_by_request_fields(monkeypatch, caplog):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    app = api.create_app(FakeDemo(), barbet_runtime=FakeBarbetRuntime())
    forged = "x\nsynthesis.request request_id=forged stage=completed"

    with caplog.at_level(logging.INFO), TestClient(app) as client:
        client.post(
            "/api/v1/synthesize",
            data={"engine_id": "voxcpm2", "model_id": forged, "text": "測試"},
        )

    received = [r.getMessage() for r in caplog.records if "stage=received" in r.getMessage()]
    assert received
    assert all("\n" not in message for message in received)
    assert "x\\nsynthesis.request" in received[0]
