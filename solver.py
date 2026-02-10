import os
import re
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    import tomllib  # py3.11+
except Exception:
    import tomli as tomllib  # pip install tomli


# =========================
# CONFIG
# =========================

SECRETS_TOML_PATH = os.environ.get("FS_SECRETS_TOML", r".streamlit\secrets.toml")
TIMEOUT_SEC = 30
VERIFY_TLS = True

# Cate station codes sa testeze la KPI RT
KPI_SAMPLE_LIMIT = 3

# Cat body sa printeze max
MAX_BODY_CHARS = 4000


# =========================
# UTIL
# =========================

def mask(s: str, keep: int = 3) -> str:
    s = "" if s is None else str(s)
    if len(s) <= keep:
        return "*" * len(s)
    return s[:keep] + "*" * (len(s) - keep)

def clip(s: str, n: int = MAX_BODY_CHARS) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n] + f"\n... (clipped, {len(s)} chars total)"

def pp_json(text: str) -> str:
    try:
        obj = json.loads(text)
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return text

def decode_fail(body: Dict[str, Any]) -> str:
    fail = body.get("failCode")
    msg = str(body.get("message", "") or "")
    low = msg.lower()

    if fail == 20400 or "user_or_value_invalid" in low:
        return "20400 LOGIN INVALID – user/system_code greșite, cont non-Northbound, cont blocat/expirat etc."
    if fail == 407 or "access_frequency_is_too_high" in low:
        return "407 RATE LIMIT – prea multe apeluri; aplică backoff/cooldown"
    if fail == 20056 or "not authorized" in low:
        return "20056 FĂRĂ DREPTURI – contul API nu e autorizat"
    return f"{fail} {msg}".strip()

def safe_headers(headers: Dict[str, str]) -> Dict[str, str]:
    h = dict(headers or {})
    for k in list(h.keys()):
        if k.lower() in {"xsrf-token", "cookie", "authorization"}:
            h[k] = mask(h[k], keep=6)
    return h

def load_secrets(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)

def get_instances(secrets: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    fs = secrets.get("fusionsolar", {}) or {}
    inst = (fs.get("instances") or {})
    out: Dict[str, Dict[str, str]] = {}

    for key, v in inst.items():
        if not isinstance(v, dict):
            continue
        out[key] = {
            "key": key,
            "label": str(v.get("label", key)),
            "base_url": str(v.get("base_url", "")).strip(),
            "username": str(v.get("username", "")).strip(),
            "system_code": str(v.get("system_code", "")).strip(),
        }
    return out


# =========================
# HTTP DEBUG HELPERS
# =========================

def post_json(session: requests.Session, base_url: str, path: str, payload: Dict[str, Any],
             xsrf_token: Optional[str] = None) -> Tuple[Dict[str, Any], requests.Response]:
    url = base_url.rstrip("/") + path

    headers = {"Content-Type": "application/json"}
    if xsrf_token:
        headers["XSRF-TOKEN"] = xsrf_token

    print("\n" + "=" * 100)
    print(f"[REQ] POST {url}")
    print(f"[REQ] headers: {safe_headers(headers)}")
    print(f"[REQ] payload: {payload}")

    t0 = time.time()
    resp = session.post(url, json=payload, headers=headers, timeout=TIMEOUT_SEC, verify=VERIFY_TLS)
    dt_ms = (time.time() - t0) * 1000

    print(f"[RES] http={resp.status_code} time_ms={dt_ms:.1f}")
    print(f"[RES] body:\n{clip(pp_json(resp.text), MAX_BODY_CHARS)}")

    resp.raise_for_status()

    try:
        body = resp.json()
    except Exception:
        raise RuntimeError(f"Răspuns non-JSON: {resp.text[:300]}")

    # cookies snapshot
    ck = session.cookies.get_dict()
    ck_safe = {k: (mask(v, keep=6) if "token" in k.lower() else v) for k, v in ck.items()}
    print(f"[COOKIES] {ck_safe}")

    return body, resp


def fs_login(session: requests.Session, base_url: str, username: str, system_code: str) -> str:
    payload = {"userName": username, "systemCode": system_code}
    body, _ = post_json(session, base_url, "/thirdData/login", payload)

    if not body.get("success") or body.get("failCode") != 0:
        raise RuntimeError(f"Login eșuat: {body} | {decode_fail(body)}")

    # token e de obicei în cookie
    token = session.cookies.get("XSRF-TOKEN") or session.cookies.get("xsrf-token")
    if not token:
        raise RuntimeError("Nu am găsit XSRF-TOKEN în cookies după login.")
    return token


def fs_get_station_list(session: requests.Session, base_url: str, token: str) -> List[Dict[str, Any]]:
    body, _ = post_json(session, base_url, "/thirdData/getStationList", {}, xsrf_token=token)
    if not body.get("success") or body.get("failCode") != 0:
        raise RuntimeError(f"getStationList eșuat: {body} | {decode_fail(body)}")

    data = body.get("data") or []
    if isinstance(data, dict):
        data = [data]
    return data


def fs_get_station_real_kpi(session: requests.Session, base_url: str, token: str, station_codes: List[str]) -> Dict[str, Any]:
    # try multi
    payload = {"stationCodes": ",".join(station_codes)}
    body, _ = post_json(session, base_url, "/thirdData/getStationRealKpi", payload, xsrf_token=token)
    if body.get("success") and body.get("failCode") == 0:
        return body

    # fallback per-statie
    out: List[Dict[str, Any]] = []
    for code in station_codes:
        body1, _ = post_json(session, base_url, "/thirdData/getStationRealKpi", {"stationCode": code}, xsrf_token=token)
        if body1.get("success") and body1.get("failCode") == 0 and body1.get("data"):
            d = body1["data"]
            if isinstance(d, list):
                out.extend(d)
            else:
                out.append(d)
    return {"success": True, "failCode": 0, "data": out}


def pick_station_codes(stations: List[Dict[str, Any]], limit: int = KPI_SAMPLE_LIMIT) -> List[str]:
    codes: List[str] = []
    for st in stations:
        code = st.get("stationCode") or st.get("plantCode") or st.get("id") or st.get("code")
        if code:
            codes.append(str(code))
        if len(codes) >= limit:
            break
    return codes


# =========================
# MAIN VERIFY
# =========================

def verify_instance(inst: Dict[str, str]) -> Dict[str, Any]:
    label = inst["label"]
    base_url = inst["base_url"]
    username = inst["username"]
    system_code = inst["system_code"]

    print("\n" + "#" * 100)
    print(f"[INSTANCE] {label} ({inst['key']})")
    print(f"  base_url={base_url}")
    print(f"  username={username}")
    print(f"  system_code={mask(system_code)}")

    session = requests.Session()

    result = {
        "key": inst["key"],
        "label": label,
        "base_url": base_url,
        "login_ok": False,
        "stations_ok": False,
        "kpi_ok": False,
        "stations_count": 0,
        "sample_station_codes": [],
        "fail": None,
    }

    try:
        token = fs_login(session, base_url, username, system_code)
        print(f"[LOGIN] OK token={mask(token, keep=8)}")
        result["login_ok"] = True

        stations = fs_get_station_list(session, base_url, token)
        result["stations_ok"] = True
        result["stations_count"] = len(stations)
        print(f"[STATIONS] OK count={len(stations)}")

        sample_codes = pick_station_codes(stations, KPI_SAMPLE_LIMIT)
        result["sample_station_codes"] = sample_codes
        print(f"[STATIONS] sample codes={sample_codes}")

        if sample_codes:
            kpi = fs_get_station_real_kpi(session, base_url, token, sample_codes)
            data = kpi.get("data") or []
            print(f"[KPI] OK rows={len(data)}")
            # afișăm cheile disponibile, utile pentru mapări
            if data:
                first = data[0]
                dim = first.get("dataItemMap", {}) if isinstance(first, dict) else {}
                print(f"[KPI] first row keys={list(first.keys())}")
                if isinstance(dim, dict):
                    print(f"[KPI] dataItemMap keys(sample)={list(dim.keys())[:30]}")
            result["kpi_ok"] = True
        else:
            print("[KPI] SKIP (nu am station codes)")

    except Exception as e:
        result["fail"] = str(e)
        print(f"[FAIL] {e}")

    return result


def main():
    secrets = load_secrets(SECRETS_TOML_PATH)
    instances = get_instances(secrets)

    if not instances:
        raise RuntimeError("Nu am găsit [fusionsolar.instances.*] în secrets.toml")

    print(f"[INFO] Found {len(instances)} instances in {SECRETS_TOML_PATH}: {list(instances.keys())}")

    summary: List[Dict[str, Any]] = []
    for key in instances.keys():
        inst = instances[key]
        # sanity check: base_url/username/system_code
        if not inst["base_url"] or not inst["username"] or not inst["system_code"]:
            print(f"[SKIP] {key} missing base_url/username/system_code")
            continue
        summary.append(verify_instance(inst))

    print("\n" + "=" * 100)
    print("SUMMARY")
    ok_login = [x for x in summary if x["login_ok"]]
    ok_st = [x for x in summary if x["stations_ok"]]
    ok_kpi = [x for x in summary if x["kpi_ok"]]
    print(f"  login_ok:   {len(ok_login)}/{len(summary)}")
    print(f"  stations_ok:{len(ok_st)}/{len(summary)}")
    print(f"  kpi_ok:     {len(ok_kpi)}/{len(summary)}")

    for x in summary:
        status = "OK" if x["kpi_ok"] else ("LOGIN_OK" if x["login_ok"] else "FAIL")
        print(f"  - {x['label']:<25} {status:<8} stations={x['stations_count']:<4} fail={x['fail']}")

if __name__ == "__main__":
    main()
