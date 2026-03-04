from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

DEFAULT_BASE_URL = "https://eu5.fusionsolar.huawei.com"

LOGIN_EP = "/thirdData/login"
PLANT_LIST_EPS = ["/thirdData/getStationList", "/thirdData/stations"]
STATION_REAL_KPI_EP = "/thirdData/getStationRealKpi"
DEV_LIST_EP = "/thirdData/getDevList"
DEV_REAL_KPI_EP = "/thirdData/getDevRealKpi"

INVERTER_DEV_TYPE_IDS = (1, 38)

ACTIVE_POWER_KEYS = ("active_power", "activePower", "active_power_kw", "active-power")

STATION_RT_POWER_KEYS = (
    "realTimePower",
    "real_time_power",
    "realtimePower",
    "realtime_power",
    "activePower",
    "active_power",
)

_AUTH_CACHE: Dict[str, Dict[str, Any]] = {}
_RATE_STATE: Dict[str, Dict[str, Any]] = {}

DEFAULT_MIN_INTERVAL_SEC = 1.2
DEFAULT_COOLDOWN_SEC = 120.0
DEFAULT_MAX_RETRIES = 3


def _auth_key(base_url: str, username: str, system_code: str) -> str:
    return f"{base_url.rstrip('/')}|{username}|{system_code}"


def _rate_key(base_url: str) -> str:
    return base_url.rstrip("/")


def _get_rate_cfg() -> Tuple[float, float, int]:
    min_interval = float(os.getenv("FUSIONSOLAR_MIN_INTERVAL_SEC", str(DEFAULT_MIN_INTERVAL_SEC)))
    cooldown = float(os.getenv("FUSIONSOLAR_COOLDOWN_SEC", str(DEFAULT_COOLDOWN_SEC)))
    retries = int(os.getenv("FUSIONSOLAR_MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))
    if min_interval < 0:
        min_interval = DEFAULT_MIN_INTERVAL_SEC
    if cooldown < 0:
        cooldown = DEFAULT_COOLDOWN_SEC
    if retries < 0:
        retries = DEFAULT_MAX_RETRIES
    return min_interval, cooldown, retries


def _set_auth(base_url: str, username: str, system_code: str, token: str) -> None:
    k = _auth_key(base_url, username, system_code)
    _AUTH_CACHE[k] = {"token": token, "ts": time.time()}


def _throttle(base_url: str) -> None:
    rk = _rate_key(base_url)
    st = _RATE_STATE.setdefault(rk, {"last_call": 0.0, "cooldown_until": 0.0})
    now = time.time()
    cd = float(st.get("cooldown_until", 0.0) or 0.0)
    if now < cd:
        raise RuntimeError(f"RATE LIMIT (407) – cooldown activ ({int(cd - now)}s)")
    min_interval, _, _ = _get_rate_cfg()
    last_call = float(st.get("last_call", 0.0) or 0.0)
    wait = (last_call + min_interval) - now
    if wait > 0:
        time.sleep(wait)
    st["last_call"] = time.time()


def load_config_from_env_or_secrets(secrets: Optional[dict] = None) -> Tuple[str, str, str]:
    secrets = secrets or {}
    base_url = secrets.get("FUSIONSOLAR_BASE_URL") or os.getenv("FUSIONSOLAR_BASE_URL") or DEFAULT_BASE_URL
    username = secrets.get("FUSIONSOLAR_API_USERNAME") or os.getenv("FUSIONSOLAR_API_USERNAME") or ""
    system_code = secrets.get("FUSIONSOLAR_API_SYSTEM_CODE") or os.getenv("FUSIONSOLAR_API_SYSTEM_CODE") or ""
    return base_url, username, system_code


def _post(
    session: requests.Session,
    base_url: str,
    endpoint: str,
    token: Optional[str],
    payload: Dict[str, Any],
) -> requests.Response:
    url = base_url.rstrip("/") + endpoint
    headers = {"Content-Type": "application/json"}
    if token:
        headers["XSRF-TOKEN"] = token
        headers["xsrf-token"] = token

    min_interval, cooldown_sec, max_retries = _get_rate_cfg()
    rk = _rate_key(base_url)
    st = _RATE_STATE.setdefault(rk, {"last_call": 0.0, "cooldown_until": 0.0})

    last_exc: Optional[BaseException] = None
    backoff = 1.0

    for attempt in range(max_retries + 1):
        _throttle(base_url)

        try:
            resp = session.post(url, headers=headers, json=payload, timeout=30)
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 10.0)
                continue
            raise

        try:
            body = resp.json() if resp.headers.get("Content-Type", "").lower().startswith("application/json") else None
        except Exception:
            body = None

        is_407 = False
        if body and isinstance(body, dict):
            fc = body.get("failCode")
            if fc == 407 or str(fc) == "407":
                is_407 = True
            msg = str(body.get("message") or body.get("failReason") or body.get("data") or "").lower()
            if "access_frequency_is_too_high" in msg:
                is_407 = True

        if resp.status_code in (429,) or is_407:
            st["cooldown_until"] = time.time() + cooldown_sec
            if attempt < max_retries:
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 10.0)
                continue
            raise RuntimeError("RATE LIMIT (407) – ACCESS_FREQUENCY_IS_TOO_HIGH")

        if resp.status_code in (500, 502, 503, 504):
            if attempt < max_retries:
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 10.0)
                continue

        return resp

    if last_exc:
        raise RuntimeError(str(last_exc))
    raise RuntimeError("Request eșuat")


def _extract_list(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("list", "records", "pageList", "dataList", "stationList", "plantList"):
            v = data.get(k)
            if isinstance(v, list):
                return v
    return []


def login(session: requests.Session, base_url: str, username: str, system_code: str) -> str:
    if not username or not system_code:
        raise RuntimeError("Lipsesc credențialele FusionSolar (username/system_code).")

    resp = _post(
        session=session,
        base_url=base_url,
        endpoint=LOGIN_EP,
        token=None,
        payload={"userName": username, "systemCode": system_code},
    )
    resp.raise_for_status()
    body = resp.json()

    if not body.get("success") or body.get("failCode") != 0:
        raise RuntimeError(f"Login eșuat: {body}")

    token = resp.headers.get("XSRF-TOKEN") or resp.headers.get("xsrf-token")
    if not token:
        raise RuntimeError("Nu am găsit XSRF-TOKEN în răspunsul de login.")

    _set_auth(base_url, username, system_code, token)
    return token


def get_plants(session: requests.Session, base_url: str, token: str, page_size: int = 200) -> List[Dict[str, Any]]:
    last_err: Any = None

    for ep in PLANT_LIST_EPS:
        try:
            resp = _post(session, base_url, ep, token=token, payload={})
            resp.raise_for_status()
            body = resp.json()
            if body.get("success") and body.get("failCode") == 0:
                return _extract_list(body.get("data"))
            last_err = body
        except Exception as e:
            last_err = e

        try:
            all_items: List[Dict[str, Any]] = []
            page_no = 1
            while True:
                resp = _post(session, base_url, ep, token=token, payload={"pageNo": page_no, "pageSize": page_size})
                resp.raise_for_status()
                body = resp.json()

                if not (body.get("success") and body.get("failCode") == 0):
                    last_err = body
                    break

                chunk = _extract_list(body.get("data"))
                all_items.extend(chunk)

                if len(chunk) < page_size:
                    break
                page_no += 1

            if all_items:
                return all_items
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Nu pot obține lista de stații. Ultima eroare: {last_err}")


def normalize_plant(row: Dict[str, Any]) -> Optional[Dict[str, str]]:
    code = row.get("stationCode") or row.get("plantCode") or row.get("id")
    if not code:
        return None

    name = row.get("stationName") or row.get("plantName") or str(code)

    status_raw = ""
    for k in (
        "stationLinkStatus",
        "linkStatus",
        "connectStatus",
        "stationStatus",
        "status",
        "communicationStatus",
        "comState",
    ):
        if row.get(k) is not None:
            status_raw = str(row.get(k))
            break

    return {"code": str(code), "name": str(name), "status_raw": status_raw}


def get_station_real_kpi(
    session: requests.Session,
    base_url: str,
    token: str,
    station_codes: List[str],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not station_codes:
        return out

    try:
        resp = _post(session, base_url, STATION_REAL_KPI_EP, token=token, payload={"stationCodes": ",".join(station_codes)})
        resp.raise_for_status()
        body = resp.json()
        if body.get("success") and body.get("data"):
            data_list = body["data"]
            if isinstance(data_list, dict):
                data_list = [data_list]
            for entry in data_list:
                code = entry.get("stationCode") or entry.get("plantCode") or entry.get("id")
                if code:
                    out[str(code)] = entry
            if out:
                return out
    except Exception:
        pass

    for code in station_codes:
        try:
            resp = _post(session, base_url, STATION_REAL_KPI_EP, token=token, payload={"stationCode": code})
            resp.raise_for_status()
            body = resp.json()
            if not (body.get("success") and body.get("data")):
                continue
            entry = body["data"][0] if isinstance(body["data"], list) else body["data"]
            out[str(code)] = entry
        except Exception:
            continue

    return out


def extract_active_power_kw(data_item_map: Dict[str, Any]) -> Optional[float]:
    if not isinstance(data_item_map, dict):
        return None
    for k in ACTIVE_POWER_KEYS:
        if k in data_item_map and data_item_map[k] is not None:
            try:
                val = float(data_item_map[k])
                if val > 100000:
                    val = val / 1000.0
                return val
            except Exception:
                return None
    return None


def extract_station_rt_power_kw(entry: Dict[str, Any]) -> Optional[float]:
    if not isinstance(entry, dict):
        return None
    dim = entry.get("dataItemMap") or {}
    if not isinstance(dim, dict):
        return None

    for k in STATION_RT_POWER_KEYS:
        if k in dim and dim[k] is not None:
            try:
                val = float(dim[k])
                if val > 100000:
                    val = val / 1000.0
                return val
            except Exception:
                return None
    return None


def get_devices_by_plants(session: requests.Session, base_url: str, token: str, plant_codes: List[str]) -> List[Dict[str, Any]]:
    devices: List[Dict[str, Any]] = []
    if not plant_codes:
        return devices

    CHUNK = 100
    for i in range(0, len(plant_codes), CHUNK):
        chunk = plant_codes[i:i + CHUNK]
        resp = _post(session, base_url, DEV_LIST_EP, token=token, payload={"stationCodes": ",".join(chunk)})
        resp.raise_for_status()
        body = resp.json()

        if not (body.get("success") and body.get("failCode") == 0):
            raise RuntimeError(f"getDevList eșuat: {body}")

        data = body.get("data") or []
        if isinstance(data, dict):
            data = _extract_list(data)
        if isinstance(data, list):
            devices.extend(data)

    return devices


def get_dev_real_kpi(
    session: requests.Session,
    base_url: str,
    token: str,
    dev_type_id: int,
    dev_ids: List[int],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not dev_ids:
        return out

    CHUNK = 100

    for i in range(0, len(dev_ids), CHUNK):
        chunk = dev_ids[i:i + CHUNK]

        payload_variants = [
            {"devTypeId": int(dev_type_id), "devIds": chunk},
            {"devTypeId": int(dev_type_id), "devIds": ",".join(map(str, chunk))},
            {"devTypeId": str(dev_type_id), "devIds": ",".join(map(str, chunk))},
            {"devTypeId": int(dev_type_id), "devIdsStr": ",".join(map(str, chunk))},
        ]

        last_err = None
        ok = False

        for payload in payload_variants:
            try:
                resp = _post(session, base_url, DEV_REAL_KPI_EP, token=token, payload=payload)
                if resp.status_code == 400:
                    raise RuntimeError(f"getDevRealKpi 400 (payload={payload}) -> {resp.text}")

                resp.raise_for_status()
                body = resp.json()

                if not (body.get("success") and body.get("failCode") == 0):
                    raise RuntimeError(f"getDevRealKpi eșuat (payload={payload}): {body}")

                data = body.get("data") or []
                if isinstance(data, dict):
                    data = [data]
                if isinstance(data, list):
                    out.extend(data)

                ok = True
                break

            except Exception as e:
                last_err = e

        if not ok:
            raise RuntimeError(str(last_err))

    return out