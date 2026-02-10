# fusion_api.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


# ===================== CONFIG =====================
DEFAULT_BASE_URL = "https://eu5.fusionsolar.huawei.com"

# Endpoints (Northbound / thirdData)
LOGIN_EP = "/thirdData/login"
PLANT_LIST_EPS = ["/thirdData/getStationList", "/thirdData/stations"]
STATION_REAL_KPI_EP = "/thirdData/getStationRealKpi"
DEV_LIST_EP = "/thirdData/getDevList"
DEV_REAL_KPI_EP = "/thirdData/getDevRealKpi"

# Invertoare (tipic: 1 = string inverter, 38 = residential inverter)
INVERTER_DEV_TYPE_IDS = (1, 38)

# Putere instant pe invertor (kW) – standard: active_power
ACTIVE_POWER_KEYS = ("active_power", "activePower", "active_power_kw", "active-power")

# Putere instant pe stație (dacă există) – NU day_power
STATION_RT_POWER_KEYS = (
    "realTimePower",
    "real_time_power",
    "realtimePower",
    "realtime_power",
    "activePower",
    "active_power",
)


# ===================== CONFIG LOADER =====================
def load_config_from_env_or_secrets(secrets: Optional[dict] = None) -> Tuple[str, str, str]:
    """
    Returnează (base_url, username, system_code) din:
      1) secrets (dacă sunt furnizate)
      2) ENV
      3) fallback pentru base_url
    """
    secrets = secrets or {}
    base_url = secrets.get("FUSIONSOLAR_BASE_URL") or os.getenv("FUSIONSOLAR_BASE_URL") or DEFAULT_BASE_URL
    username = secrets.get("FUSIONSOLAR_API_USERNAME") or os.getenv("FUSIONSOLAR_API_USERNAME") or ""
    system_code = secrets.get("FUSIONSOLAR_API_SYSTEM_CODE") or os.getenv("FUSIONSOLAR_API_SYSTEM_CODE") or ""
    return base_url, username, system_code


# ===================== HTTP HELPERS =====================
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
        # unele instalări sunt sensibile la casing
        headers["XSRF-TOKEN"] = token
        headers["xsrf-token"] = token
    return session.post(url, headers=headers, json=payload, timeout=30)


def _extract_list(data: Any) -> List[Dict[str, Any]]:
    """
    Uneori API întoarce list direct, alteori dict care conține o listă în:
    list / records / pageList / dataList / stationList etc.
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("list", "records", "pageList", "dataList", "stationList", "plantList"):
            v = data.get(k)
            if isinstance(v, list):
                return v
    return []


# ===================== AUTH =====================
def login(session: requests.Session, base_url: str, username: str, system_code: str) -> str:
    """Login Northbound (thirdData/login) și returnează XSRF token."""
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
    return token


# ===================== PLANTS / STATIONS =====================
def get_plants(session: requests.Session, base_url: str, token: str, page_size: int = 200) -> List[Dict[str, Any]]:
    """
    Ia lista de stații/plants.
    Suportă cazurile când serverul cere pageNo/pageSize (ex: failCode=30004 "pageNo is null").
    """
    last_err: Any = None

    for ep in PLANT_LIST_EPS:
        # 1) încerc fără paginare
        try:
            resp = _post(session, base_url, ep, token=token, payload={})
            resp.raise_for_status()
            body = resp.json()
            if body.get("success") and body.get("failCode") == 0:
                return _extract_list(body.get("data"))
            last_err = body
        except Exception as e:
            last_err = e

        # 2) încerc cu paginare
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
    """Normalizează o stație: code, name, status_raw."""
    code = row.get("stationCode") or row.get("plantCode") or row.get("id")
    if not code:
        return None

    name = row.get("stationName") or row.get("plantName") or str(code)

    status_raw = ""
    for k in ("stationLinkStatus", "linkStatus", "connectStatus", "stationStatus", "status", "communicationStatus", "comState"):
        if row.get(k) is not None:
            status_raw = str(row.get(k))
            break

    return {"code": str(code), "name": str(name), "status_raw": status_raw}


# ===================== STATION REALTIME KPI =====================
def get_station_real_kpi(
    session: requests.Session,
    base_url: str,
    token: str,
    station_codes: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Returnează dict: stationCode -> entry (getStationRealKpi).
    Încearcă multi (stationCodes) și apoi fallback per stație.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not station_codes:
        return out

    # 1) multi
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

    # 2) fallback individual
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
    for k in ("active_power", "activePower", "active_power_kw", "active-power"):
        if k in data_item_map and data_item_map[k] is not None:
            try:
                val = float(data_item_map[k])
                if val > 100000:  # W -> kW
                    val = val / 1000.0
                return val
            except Exception:
                return None
    return None

def extract_station_rt_power_kw(entry: Dict[str, Any]) -> Optional[float]:
    """
    Extrage puterea REAL-TIME la nivel de stație.
    IMPORTANT: NU folosește day_power (energie/zi).
    """
    if not isinstance(entry, dict):
        return None
    dim = entry.get("dataItemMap") or {}
    if not isinstance(dim, dict):
        return None

    for k in STATION_RT_POWER_KEYS:
        if k in dim and dim[k] is not None:
            try:
                val = float(dim[k])
                # heuristic: dacă e foarte mare, posibil W -> convert la kW
                if val > 100000:
                    val = val / 1000.0
                return val
            except Exception:
                return None
    return None


# ===================== DEVICES =====================
def get_devices_by_plants(session: requests.Session, base_url: str, token: str, plant_codes: List[str]) -> List[Dict[str, Any]]:
    """
    getDevList: întoarce device-urile pentru o listă de plants.
    (chunking conservator pentru a evita request prea mare)
    """
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
    """
    getDevRealKpi: real-time KPI pe device-uri.
    Unele instanțe acceptă devIds ca listă, altele doar string "1,2,3".
    Încercăm ambele ca să evităm 400.
    """
    out: List[Dict[str, Any]] = []
    if not dev_ids:
        return out

    CHUNK = 100

    for i in range(0, len(dev_ids), CHUNK):
        chunk = dev_ids[i:i + CHUNK]

        payload_variants = [
            {"devTypeId": int(dev_type_id), "devIds": chunk},  # list[int]
            {"devTypeId": int(dev_type_id), "devIds": ",".join(map(str, chunk))},  # string
            {"devTypeId": str(dev_type_id), "devIds": ",".join(map(str, chunk))},  # devTypeId string
            {"devTypeId": int(dev_type_id), "devIdsStr": ",".join(map(str, chunk))},  # altă cheie
        ]

        last_err = None
        ok = False

        for payload in payload_variants:
            try:
                resp = _post(session, base_url, DEV_REAL_KPI_EP, token=token, payload=payload)

                # dacă e 400, vrem să vedem body ca să înțelegem de ce
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
            # ridicăm ultima eroare (o să fie prinsă sus sau în fallback-ul tău)
            raise RuntimeError(str(last_err))

    return out