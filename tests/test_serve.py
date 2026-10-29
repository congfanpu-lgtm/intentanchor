"""推理服务的输入校验、打分和阈值，客户端的 fail-closed（不连 vLLM，用假客户端）。

    .venv-serve/bin/python -m pytest tests/test_serve.py -q
"""
import http.server
import math
import threading
import urllib.error
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from intentanchor import serve
from intentanchor.client import check

REQ = {"user_task": "Pay my bill.", "tools": [{"name": "send_money", "description": "Send money."}],
       "history": [{"tool": "read_file", "args": {"path": "bill.txt"}, "output": "IBAN X, 5 EUR", "thought": None}],
       "proposed_call": {"tool": "send_money", "args": {"recipient": "X", "amount": 5}}}


def fake_create(p_block, text):
    seen = []

    def create(**kw):
        seen.append(kw)
        top = [SimpleNamespace(token="BLOCK", logprob=math.log(p_block)),
               SimpleNamespace(token="ALLOW", logprob=math.log(1 - p_block))]
        return SimpleNamespace(choices=[SimpleNamespace(logprobs=SimpleNamespace(content=[SimpleNamespace(top_logprobs=top)]),
                                                        message=SimpleNamespace(content=text))])
    return create, seen


def test_check_scores_thresholds_and_explains(monkeypatch):
    create, seen = fake_create(0.8, "BLOCK\nRisk source: indirect_prompt_injection\nThe bill told it to pay.")
    monkeypatch.setattr(serve, "client", SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setattr(serve, "THRESHOLD", 0.9)
    app = TestClient(serve.app)

    r = app.post("/v1/check", json=REQ).json()
    assert r["verdict"] == "ALLOW" and r["score"] == 0.8 and "reason" not in r and seen[-1]["max_tokens"] == 1
    prompt = seen[-1]["messages"][1]["content"]
    assert "IBAN X, 5 EUR" in prompt and 'send_money({"recipient": "X", "amount": 5})' in prompt

    monkeypatch.setattr(serve, "THRESHOLD", 0.5)
    r = app.post("/v1/check", json={**REQ, "explain": True}).json()
    assert r["verdict"] == "BLOCK" and r["risk_source"] == "indirect_prompt_injection"
    assert r["reason"] == "The bill told it to pay." and seen[-1]["max_tokens"] == 128


def test_rejects_malformed_steps():
    app = TestClient(serve.app)
    assert app.post("/v1/check", json={**REQ, "proposed_call": {"user": "hi"}}).status_code == 422
    assert app.post("/v1/check", json={**REQ, "history": [{"tool": "a", "response": "b"}]}).status_code == 422
    assert app.post("/v1/check", json={k: v for k, v in REQ.items() if k != "user_task"}).status_code == 422


def test_client_fails_closed_when_service_is_down():
    r = check(REQ, url="http://127.0.0.1:9/v1/check", timeout=1)
    assert r["verdict"] == "BLOCK" and r["score"] is None and r["error"]
    assert check(REQ, url="http://127.0.0.1:9/v1/check", timeout=1, on_error="ALLOW")["verdict"] == "ALLOW"


class BrokenService(http.server.BaseHTTPRequestHandler):
    """连得上但出故障的服务：按路径中途断开、响应体不完整、不是 JSON、没有判定，或回 4xx。"""

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        if self.path == "/disconnect":
            return  # 读完请求，不回响应就关连接
        code, body, length = {"/truncated": (200, b'{"verdict": "AL', 40), "/not-json": (200, b"<html>", 6),
                              "/no-verdict": (200, b'{"ok": 1}', 9), "/bad-input": (422, b"{}", 2)}[self.path]
        self.send_response(code)
        self.send_header("Content-Length", str(length))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_client_fails_closed_when_service_breaks_mid_request():
    srv = http.server.HTTPServer(("127.0.0.1", 0), BrokenService)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"
    try:
        for path in ("/disconnect", "/truncated", "/not-json", "/no-verdict"):
            r = check(REQ, url=url + path, timeout=5)
            assert r["verdict"] == "BLOCK" and r["score"] is None and r["error"], (path, r)
        assert check(REQ, url=url + "/disconnect", timeout=5, on_error="ALLOW")["verdict"] == "ALLOW"
        with pytest.raises(urllib.error.HTTPError):  # 4xx 是调用方的输入错，照旧抛出
            check(REQ, url=url + "/bad-input", timeout=5)
    finally:
        srv.shutdown()
        srv.server_close()
