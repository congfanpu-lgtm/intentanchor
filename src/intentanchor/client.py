"""守卫服务的 Python 客户端（设计文档 §5.6），只用标准库。agent 执行有副作用的工具调用前调用一次：

    from intentanchor.client import check
    r = check({"user_task": ..., "tools": [...], "history": [...], "proposed_call": {...}})
    if r["verdict"] == "BLOCK": ...  # 不执行，告诉 agent 这一步被拦了

服务出故障（连不上、超时、5xx、中途断开、响应不完整、不是 JSON 或没有合法判定）时返回 on_error 指定的判定，
默认 BLOCK（fail-closed），结果里带 "error"。
只读工具本来就不送检（§4.4），所以 fail-closed 不影响它们。4xx 是输入格式错，直接抛出，不替调用方做决定。
"""
import http.client
import json
import urllib.error
import urllib.request


def check(inp: dict, url: str = "http://localhost:18080/v1/check", explain: bool = False, timeout: float = 5.0,
          on_error: str = "BLOCK") -> dict:
    req = urllib.request.Request(url, json.dumps({**inp, "explain": explain}).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.load(r)
    except urllib.error.HTTPError as e:  # HTTPError 也是 OSError，要先接住
        if e.code < 500:
            raise
        return {"verdict": on_error, "score": None, "error": f"HTTP {e.code}"}
    # 连不上、超时、中途断开是 OSError；响应不完整是 HTTPException；不是 JSON 是 ValueError
    except (OSError, http.client.HTTPException, ValueError) as e:
        return {"verdict": on_error, "score": None, "error": f"{type(e).__name__}: {e}"}
    if not isinstance(out, dict) or out.get("verdict") not in ("ALLOW", "BLOCK"):
        return {"verdict": on_error, "score": None, "error": f"no verdict in response: {str(out)[:200]}"}
    return out
