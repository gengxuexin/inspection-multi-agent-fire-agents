# tests/test_auth_audit.py
"""common/auth.py 与 common/audit.py 的单元测试。

运行（项目根目录）：
    python -m pytest tests/test_auth_audit.py -v
"""
import json
import os
import sys

# 保证可从项目根导入 common 包（无论从哪个目录启动 pytest）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import pytest
except ImportError as exc:  # pragma: no cover
    raise ImportError("缺少 pytest，请先安装：pip install pytest") from exc

try:
    import jwt  # noqa: F401  PyJWT 可用性检查
    from fastapi import Body, Depends, FastAPI
    from fastapi.testclient import TestClient

    from common.auth import (
        UserCtx,
        create_token,
        get_current_user,
        require_internal_token,
        verify_token,
    )
    from common.audit import audit_event, install_audit_middleware, redact
    from common.config import get_settings
except ImportError:
    pytest.skip("依赖未安装", allow_module_level=True)


# ---------------- 环境隔离 ----------------


@pytest.fixture()
def auth_env(monkeypatch):
    """设置 JWT 相关环境变量并清理 get_settings 缓存（teardown 再清一次）。"""
    monkeypatch.setenv("JWT_SECRET", "unit-test-secret")
    monkeypatch.setenv("JWT_ISSUER", "test-issuer")
    monkeypatch.setenv("JWT_AUDIENCE", "test-audience")
    monkeypatch.setenv("JWT_EXPIRE_MIN", "60")
    monkeypatch.setenv("JWT_OIDC_JWKS_URL", "")
    monkeypatch.setenv("AUTH_DISABLED", "false")
    monkeypatch.setenv("AGENT_SHARED_TOKEN", "")
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


def _build_app(log_path):
    """迷你 FastAPI 应用：受保护路由 + JSON 回显 + 审计中间件。"""
    app = FastAPI()

    @app.get("/secure")
    async def secure(user: UserCtx = Depends(get_current_user)):
        return {"sub": user.sub, "dept": user.dept, "clearance": user.clearance}

    @app.post("/echo")
    async def echo(payload: dict = Body(...), user: UserCtx = Depends(get_current_user)):
        return {"got": payload}

    @app.get("/")
    async def health():
        return {"ok": True}

    install_audit_middleware(app, log_path=str(log_path))
    return app


# ---------------- redact ----------------


def test_redact_phone():
    out = redact("联系人 13812345678，备用 15987654321")
    assert "138****5678" in out
    assert "159****4321" in out
    assert "13812345678" not in out
    assert "15987654321" not in out


def test_redact_idcard():
    out = redact("身份证：11010119900307777X")
    assert "110101********777X" in out
    assert "19900307" not in out


def test_redact_secret_kv():
    out = redact("password=hunter2 api_key=abc123 TOKEN:zzz")
    assert "hunter2" not in out and "password=***" in out
    assert "abc123" not in out and "api_key=***" in out
    assert "TOKEN=***" in out


def test_redact_sk_key():
    out = redact("使用密钥 sk-abcdef1234567890xyz 调用")
    assert "sk-***" in out
    assert "abcdef1234567890" not in out


def test_redact_truncate_and_empty():
    out = redact("a" * 1000, max_len=100)
    assert out == "a" * 100 + "...(截断)"
    assert redact("短文本") == "短文本"
    assert redact(None) == ""
    assert redact("") == ""


# ---------------- JWT 往返 ----------------


def test_create_and_verify_token(auth_env):
    token = create_token("officer-1", dept="特勤大队", role="chief", clearance=3)
    claims = verify_token(token)
    assert claims["sub"] == "officer-1"
    assert claims["dept"] == "特勤大队"
    assert claims["role"] == "chief"
    assert claims["clearance"] == 3
    assert claims["iss"] == "test-issuer"
    assert claims["aud"] == "test-audience"


def test_tampered_token_raises_valueerror(auth_env):
    token = create_token("officer-2")
    bad = token[:-3] + ("aaa" if not token.endswith("aaa") else "bbb")
    with pytest.raises(ValueError):
        verify_token(bad)


def test_create_token_requires_secret(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "")
    get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError):
            create_token("someone")
    finally:
        get_settings.cache_clear()


# ---------------- FastAPI 集成 ----------------


def test_no_token_401(auth_env, tmp_path):
    client = TestClient(_build_app(tmp_path / "audit.jsonl"))
    r = client.get("/secure")
    assert r.status_code == 401
    assert "detail" in r.json()


def test_exempt_path_and_valid_token(auth_env, tmp_path):
    client = TestClient(_build_app(tmp_path / "audit.jsonl"))
    # 豁免路径无需 token
    assert client.get("/").status_code == 200
    # 有效 token 放行
    token = create_token("zhang-san")
    r = client.get("/secure", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["sub"] == "zhang-san"
    assert r.json()["dept"] == "消防支队"


def test_auth_disabled_bypass(auth_env, monkeypatch, tmp_path):
    monkeypatch.setenv("AUTH_DISABLED", "true")
    get_settings.cache_clear()
    client = TestClient(_build_app(tmp_path / "audit.jsonl"))
    r = client.get("/secure")
    assert r.status_code == 200
    assert r.json()["sub"] == "dev"


def test_audit_jsonl_written_and_redacted(auth_env, tmp_path):
    log_path = tmp_path / "audit.jsonl"
    client = TestClient(_build_app(log_path))
    token = create_token("zhang-san")

    r = client.post(
        "/echo",
        headers={"Authorization": f"Bearer {token}"},
        json={"phone": "13812345678"},
    )
    assert r.status_code == 200

    assert log_path.exists(), "audit.jsonl 应被写出"
    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records, "audit.jsonl 应至少有一条记录"

    echo_rec = next(x for x in records if x["path"] == "/echo")
    assert echo_rec["sub"] == "zhang-san"
    assert echo_rec["method"] == "POST"
    assert "138****5678" in echo_rec["input_summary"]
    assert echo_rec["output_summary"], "JSON 响应应被捕获"
    assert "138****5678" in echo_rec["output_summary"]
    # 整份记录中不得出现未脱敏手机号
    assert "13812345678" not in json.dumps(records, ensure_ascii=False)
    for field in (
        "ts", "sub", "method", "path", "status",
        "duration_ms", "client", "input_summary", "output_summary",
    ):
        assert field in echo_rec


def test_require_internal_token(auth_env, monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_SHARED_TOKEN", "inner-secret")
    get_settings.cache_clear()
    app = FastAPI()

    @app.get("/")
    async def health():
        return {"ok": True}

    @app.get("/internal/task", dependencies=[Depends(require_internal_token)])
    async def internal_task():
        return {"ok": True}

    install_audit_middleware(app, log_path=str(tmp_path / "audit.jsonl"))
    client = TestClient(app)
    # 根路径（健康检查）放行
    assert client.get("/").status_code == 200
    # 缺内部 token → 401
    assert client.get("/internal/task").status_code == 401
    # 正确内部 token → 200
    ok = client.get("/internal/task", headers={"X-Internal-Token": "inner-secret"})
    assert ok.status_code == 200


def test_audit_event(auth_env, tmp_path):
    p = tmp_path / "audit.jsonl"
    audit_event(
        "review_approve",
        user=UserCtx(sub="lisi"),
        log_path=str(p),
        case_id="C-001",
        note="当事人 13812345678 未整改",
    )
    rec = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
    assert rec["action"] == "review_approve"
    assert rec["sub"] == "lisi"
    assert "138****5678" in rec["note"]
    assert "13812345678" not in rec["note"]