"""這個服務念哪個語言，以及跟請求的 lang 對不對得上。

同一份程式跑成兩個容器：台語（app，預設）與華語原版（app-zh），由 compose
的 VOXCPM_LANG 區分。nginx 只看網址的 ?lang= 決定送去哪個容器，讀不到表單
與 JSON；呼叫端若把 lang 寫在 body，請求會被送到台語服務、靜默念成台語。
所以 body 帶 lang 時由這裡檢查，對不上就回 400，而不是默默念錯語言。
"""

import json
import os
from typing import Any
from urllib.parse import parse_qs

from fastapi import HTTPException

SERVICE_LANGS = ("tw", "zh")
DEFAULT_SERVICE_LANG = "tw"
LANG_HEADER = "X-TTS-Lang"


def service_lang() -> str:
    """每次都讀環境變數，測試才能逐案切換。"""
    lang = os.environ.get("VOXCPM_LANG", "").strip() or DEFAULT_SERVICE_LANG
    if lang not in SERVICE_LANGS:
        raise RuntimeError(
            f"VOXCPM_LANG 只接受 {' 或 '.join(SERVICE_LANGS)}，收到 {lang!r}"
        )
    return lang


def with_service_lang(path: str) -> str:
    """回給呼叫端的 API 網址要帶回 lang，後續輪詢才會被送回同一個容器。"""
    lang = service_lang()
    if lang == DEFAULT_SERVICE_LANG:
        return path
    separator = "&" if "?" in path else "?"
    return f"{path}{separator}lang={lang}"


def voice_language(language: str) -> str:
    """CastVoice 聲音清單的語言欄位：華語服務念的是華語，不是聲音原本的台語。"""
    return "zh-TW" if service_lang() == "zh" else language


def voice_label(label: str) -> str:
    if service_lang() != "zh":
        return label
    return label.replace("臺灣台語", "華語")


def check_body_lang(value: Any) -> None:
    """body 裡的 lang 只能跟本服務一致；沒帶就略過。"""
    # 測試直接呼叫路由函式時拿到的是 Form 預設物件而非字串
    if not isinstance(value, str) or not value.strip():
        return
    requested = value.strip().lower()
    lang = service_lang()
    if requested == lang:
        return
    if requested not in SERVICE_LANGS:
        detail = f"lang 只接受 {' 或 '.join(SERVICE_LANGS)}"
    else:
        detail = (
            f"這個服務念的是 {lang}，請把語言放在網址：?lang={requested}"
            "（只寫在表單或 JSON 裡不會切換服務）"
        )
    raise HTTPException(status_code=400, detail=detail)


class ServiceLangMiddleware:
    """純 ASGI middleware：檢查網址的 lang，並在每個回應標上本服務語言。

    不用 BaseHTTPMiddleware：它會改變串流與 client 中斷時的取消行為，
    而 GPU gate 依賴「中斷後仍持有 gate 到工作結束」的語意。
    """

    def __init__(self, app: Any) -> None:
        self.app = app
        self.lang = service_lang()

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # 跟 nginx 一致：取第一個 lang（空值也算）、不分大小寫，兩邊才不會各認各的。
        query = parse_qs(
            scope.get("query_string", b"").decode("latin-1"),
            keep_blank_values=True,
        )
        requested = (query.get("lang") or [""])[0].strip().lower()
        if requested and requested != self.lang:
            body = json.dumps(
                {"detail": f"這個服務念的是 {self.lang}，lang={requested} 的請求送錯服務"},
                ensure_ascii=False,
            ).encode("utf-8")
            await send({
                "type": "http.response.start",
                "status": 400,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (LANG_HEADER.lower().encode(), self.lang.encode()),
                ],
            })
            await send({"type": "http.response.body", "body": body})
            return

        header = (LANG_HEADER.lower().encode(), self.lang.encode())

        async def send_with_lang(message: dict) -> None:
            if message["type"] == "http.response.start":
                message = {**message, "headers": [*message.get("headers", []), header]}
            await send(message)

        await self.app(scope, receive, send_with_lang)
