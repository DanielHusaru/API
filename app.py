import time
import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from supabase import create_client
import fusion_api as fs  

FS_TZ = ZoneInfo("Europe/Bucharest")
FS_SUM_EXCLUDE_NAME_CONTAINS = ["raal", "transavia", "aldgate"]

SB_TABLE_MAIN = "fs_power_snapshots"       
SB_TABLE_SCRAPE = "fs_power_snapshots_rt"  

RT_FORCE_TS_SHIFT_HOURS: Optional[int] = None
FUTURE_TOL_MIN = 2
PAGE_SIZE = 1000
MAX_PAGES = 200  

def _now_local_str() -> str:
    return datetime.datetime.now(tz=FS_TZ).strftime("%Y-%m-%d %H:%M:%S")

def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)

def _today_local_date() -> datetime.date:
    return datetime.datetime.now(FS_TZ).date()

def _norm_name(s: str) -> str:
    return " ".join(str(s or "").strip().split())

def _norm_name_key(s: str) -> str:
    return _norm_name(s).lower()

def _is_excluded_name(name: str) -> bool:
    low = str(name or "").lower()
    return any(x in low for x in FS_SUM_EXCLUDE_NAME_CONTAINS)

def _decode_error_reason(msg: str) -> str:
    low = (msg or "").lower()
    if "failcode" in low and "407" in low:
        return "RATE LIMIT (407) – ACCESS_FREQUENCY_IS_TOO_HIGH"
    if "20400" in low or "user_or_value_invalid" in low:
        return "LOGIN INVALID (20400) – user/system_code greșite sau nu e cont Northbound"
    if "20056" in low and ("not authorized" in low or "not authorized by the owner" in low):
        return "FĂRĂ DREPTURI (20056) – contul API nu e autorizat"
    return msg

def _apply_aliases_inplace(df: pd.DataFrame) -> pd.DataFrame:
    cfg = st.secrets.get("fusionsolar", {}) or {}
    aliases_by_key_raw = dict(cfg.get("aliases_by_key", {}) or {})
    aliases_by_name_raw = dict(cfg.get("aliases_by_name", {}) or {})

    if df is None or df.empty or "Nume" not in df.columns:
        return df

    aliases_by_name = {_norm_name_key(k): v for k, v in aliases_by_name_raw.items()}
    aliases_by_key: Dict[str, str] = {}
    for k, v in aliases_by_key_raw.items():
        kk = str(k)
        if "|" in kk:
            inst, name = kk.split("|", 1)
            aliases_by_key[f"{_norm_name_key(inst)}|{_norm_name_key(name)}"] = v
        else:
            aliases_by_key[_norm_name_key(kk)] = v

    df2 = df.copy()
    inst_col = df2["Instanță"].astype(str).fillna("") if "Instanță" in df2.columns else pd.Series([""] * len(df2))
    name_col = df2["Nume"].astype(str).fillna("")
    inst_norm = inst_col.map(_norm_name_key)
    name_norm = name_col.map(_norm_name_key)
    key_norm = inst_norm + "|" + name_norm

    mapped = key_norm.map(aliases_by_key)
    if mapped.isna().all():
        mapped = pd.Series([None] * len(df2), index=df2.index)

    df2["Nume"] = mapped.fillna(name_norm.map(aliases_by_name)).fillna(name_col)
    return df2

def _load_instances_from_secrets() -> Tuple[str, int, Dict[str, dict]]:
    cfg = st.secrets.get("fusionsolar", {}) or {}
    title = str(cfg.get("title", "Measurement APIs"))
    token_ttl_sec = int(cfg.get("token_ttl_sec", 1800))
    instances_raw = cfg.get("instances", {})
    if not instances_raw:
        raise RuntimeError("Nu există instanțe în secrets.toml.")

    instances: Dict[str, dict] = {}
    for key, inst in instances_raw.items():
        base_url = str(inst.get("base_url", "")).strip() or fs.DEFAULT_BASE_URL
        username = str(inst.get("username", "")).strip()
        system_code = str(inst.get("system_code", "")).strip()
        label = str(inst.get("label", key)).strip()
        manual_stations = inst.get("stations") or []
        parsed_manual: List[dict] = []
        if isinstance(manual_stations, list):
            for s in manual_stations:
                if not isinstance(s, dict):
                    continue
                c = str(s.get("code", "")).strip()
                n = str(s.get("name", c)).strip()
                if c:
                    parsed_manual.append({"code": c, "name": n})
        if not username or not system_code:
            continue
        instances[key] = {
            "key": key,
            "label": label,
            "base_url": base_url,
            "username": username,
            "system_code": system_code,
            "manual_stations": parsed_manual,
        }
    if not instances:
        raise RuntimeError("Instanțele sunt incomplete.")
    return title, token_ttl_sec, instances

def _token_get_or_login(session: requests.Session, inst_key: str, inst: dict, ttl_sec: int) -> str:
    token_key = f"fs_token_{inst_key}"
    ts_key = f"fs_token_ts_{inst_key}"
    cooldown_key = f"fs_cooldown_until_{inst_key}"
    last_try_key = f"fs_login_try_{inst_key}"
    now = time.time()
    cooldown_until = float(st.session_state.get(cooldown_key, 0))
    if now < cooldown_until:
        raise RuntimeError(f"RATE LIMIT (407) – cooldown activ ({int(cooldown_until - now)}s)")
    need_login = (
        token_key not in st.session_state
        or ts_key not in st.session_state
        or (now - float(st.session_state.get(ts_key, 0))) > ttl_sec
    )
    if not need_login:
        return st.session_state[token_key]
    last_try = float(st.session_state.get(last_try_key, 0))
    if (now - last_try) < 60:
        raise RuntimeError("LOGIN RATE-LIMIT: așteaptă ~60s")
    st.session_state[last_try_key] = now
    delays = [0.0, 1.0, 2.5]
    last_exc = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            token = fs.login(session, inst["base_url"], inst["username"], inst["system_code"])
            st.session_state[token_key] = token
            st.session_state[ts_key] = time.time()
            return token
        except Exception as e:
            last_exc = e
            msg = str(e).lower()
            if "access_frequency_is_too_high" in msg or ("failcode" in msg and "407" in msg):
                st.session_state[cooldown_key] = time.time() + 120
                raise RuntimeError("RATE LIMIT (407) – cooldown 120s") from e
    raise RuntimeError(f"Login eșuat: {last_exc}")

def _compute_active(status_raw: str, power_kw: Optional[float]) -> Tuple[bool, str]:
    offline_markers = {"0", "offline", "disconnected", "false", "down"}
    if status_raw and str(status_raw).strip().lower() in offline_markers:
        return False, "OFFLINE"
    if power_kw is None:
        return False, "FĂRĂ KPI (active_power)"
    return True, "OK"

def _fetch_table_all_instances(instances: Dict[str, dict], token_ttl_sec: int) -> pd.DataFrame:
    if "fs_http" not in st.session_state:
        st.session_state["fs_http"] = requests.Session()
    session: requests.Session = st.session_state["fs_http"]
    rows: List[Dict[str, Any]] = []
    inst_keys = list(instances.keys())
    prog = st.progress(0.0)
    for idx, inst_key in enumerate(inst_keys, start=1):
        inst = instances[inst_key]
        label = inst.get("label", inst_key)
        try:
            token = _token_get_or_login(session, inst_key, inst, ttl_sec=token_ttl_sec)
            infos: List[dict] = []
            try:
                plants_raw = fs.get_plants(session, inst["base_url"], token)
                for r in plants_raw:
                    norm = fs.normalize_plant(r)
                    if norm:
                        infos.append(norm)
            except Exception as e_list:
                manual = inst.get("manual_stations") or []
                if manual:
                    infos = [{"code": s["code"], "name": s["name"], "status_raw": ""} for s in manual]
                else:
                    raise e_list
            if not infos:
                rows.append({"Instanță": label, "Nume": "", "Active": False, "Putere_RT_kW": None, "Cod": "", "Motiv": "LISTĂ GOALĂ"})
                prog.progress(idx / max(len(inst_keys), 1))
                continue
            plant_codes = [x["code"] for x in infos]
            station_kpi_map = fs.get_station_real_kpi(session, inst["base_url"], token, plant_codes)
            station_power_by_code: Dict[str, float] = {}
            missing_for_device: List[str] = []
            for code in plant_codes:
                entry = station_kpi_map.get(code, {})
                p_station = fs.extract_station_rt_power_kw(entry) if entry else None
                if p_station is None:
                    missing_for_device.append(code)
                else:
                    station_power_by_code[code] = float(p_station)
            power_by_plant: Dict[str, float] = {}
            dev_fallback_error: Optional[str] = None
            if missing_for_device:
                try:
                    devices = fs.get_devices_by_plants(session, inst["base_url"], token, missing_for_device)
                    dev_to_plant: Dict[int, str] = {}
                    dev_ids_by_type: Dict[int, List[int]] = {1: [], 38: []}
                    for d in devices:
                        try:
                            dev_id = int(d.get("id"))
                        except Exception:
                            continue
                        plant_code = str(d.get("stationCode") or d.get("station_code") or d.get("plantCode") or d.get("plant_code") or "").strip()
                        if not plant_code:
                            continue
                        try:
                            dev_type_int = int(d.get("devTypeId"))
                        except Exception:
                            continue
                        if dev_type_int not in fs.INVERTER_DEV_TYPE_IDS:
                            continue
                        dev_to_plant[dev_id] = plant_code
                        dev_ids_by_type[dev_type_int].append(dev_id)
                    for dev_type_id, dev_ids in dev_ids_by_type.items():
                        if not dev_ids:
                            continue
                        kpi_rows = fs.get_dev_real_kpi(session, inst["base_url"], token, dev_type_id, dev_ids)
                        for kr in kpi_rows:
                            dev_id_raw = kr.get("devId", kr.get("id"))
                            try:
                                dev_id = int(dev_id_raw)
                            except Exception:
                                continue
                            dim = kr.get("dataItemMap") or {}
                            p_kw = fs.extract_active_power_kw(dim)
                            if p_kw is None:
                                continue
                            plant_code = dev_to_plant.get(dev_id)
                            if not plant_code:
                                continue
                            power_by_plant[plant_code] = power_by_plant.get(plant_code, 0.0) + float(p_kw)
                except Exception as e:
                    dev_fallback_error = str(e)
            for inf in infos:
                code = inf["code"]
                name = inf["name"]
                status_raw = inf.get("status_raw", "")
                p_kw = station_power_by_code.get(code, power_by_plant.get(code))
                active, reason = _compute_active(status_raw, p_kw)
                if p_kw is None and dev_fallback_error and code in missing_for_device:
                    reason = f"FĂRĂ RT (dev fallback eșuat: {dev_fallback_error})"
                rows.append({"Instanță": label, "Nume": name, "Active": bool(active), "Putere_RT_kW": p_kw, "Cod": code, "Motiv": reason})
        except Exception as e:
            rows.append({"Instanță": label, "Nume": "(eroare instanță)", "Active": False, "Putere_RT_kW": None, "Cod": "", "Motiv": _decode_error_reason(str(e))})
        prog.progress(idx / max(len(inst_keys), 1))
    df = pd.DataFrame(rows)
    for col in ["Instanță", "Nume", "Active", "Putere_RT_kW", "Motiv", "Cod"]:
        if col not in df.columns:
            df[col] = None
    if not df.empty:
        df = df.sort_values(by=["Instanță", "Active", "Nume"], ascending=[True, False, True]).reset_index(drop=True)
    return df

def _sb_client(service: bool = False):
    cfg = st.secrets.get("supabase", {}) or {}
    url = str(cfg.get("url", "")).strip()
    key = str(cfg.get("service_role_key" if service else "anon_key", "")).strip()
    if not url or not key:
        raise RuntimeError("Lipsesc cheile Supabase.")
    return create_client(url, key)

def _sb_fetch_paged(table: str, select_cols: str, filter_col: str, since_iso: str, order_col: str, desc: bool) -> pd.DataFrame:
    sb = _sb_client(service=False)
    all_rows: List[dict] = []
    for page in range(MAX_PAGES):
        start = page * PAGE_SIZE
        end = start + PAGE_SIZE - 1
        q = (
            sb.table(table)
            .select(select_cols)
            .gte(filter_col, since_iso)
            .order(order_col, desc=desc)
            .range(start, end)
        )
        res = q.execute()
        rows = res.data or []
        all_rows.extend(rows)
        if len(rows) < PAGE_SIZE:
            break
    return pd.DataFrame(all_rows)

def _sb_upsert_snapshot_main(df_ins: pd.DataFrame) -> None:
    if df_ins is None or df_ins.empty:
        return
    sb = _sb_client(service=True)
    df2 = df_ins.copy()
    df2["ts_utc"] = pd.to_datetime(df2.get("ts_utc", _now_utc()), utc=True, errors="coerce")
    df2["ts_utc"] = df2["ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    for col in ["instance_key", "plant_code", "plant_name", "alias_name"]:
        if col not in df2.columns:
            df2[col] = ""
        df2[col] = df2[col].astype(str).fillna("")
    df2["power_kw"] = pd.to_numeric(df2.get("power_kw", 0.0), errors="coerce").fillna(0.0).astype(float)
    df2["station_key"] = (df2["instance_key"] + "|" + df2["plant_code"]).astype(str)
    df2 = df2[df2["plant_code"].astype(str).str.len() > 0].copy()
    df2 = df2.drop_duplicates(subset=["ts_utc", "station_key"], keep="last")
    recs = df2[["ts_utc", "station_key", "instance_key", "plant_code", "plant_name", "alias_name", "power_kw"]].to_dict("records")
    if recs:
        sb.table(SB_TABLE_MAIN).upsert(recs, on_conflict="ts_utc,station_key").execute()

def _sb_load_main_last_hours(hours_back: int = 24) -> pd.DataFrame:
    sb = _sb_client(service=False)
    since_utc = (_now_utc() - datetime.timedelta(hours=hours_back)).isoformat()
    res = (
        sb.table(SB_TABLE_MAIN)
        .select("ts_utc,station_key,instance_key,plant_code,plant_name,alias_name,power_kw")
        .gte("ts_utc", since_utc)
        .order("ts_utc", desc=False)
        .execute()
    )
    rows = res.data or []
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    df["ts_local"] = df["ts_utc"].dt.tz_convert(FS_TZ)
    df["power_kw"] = pd.to_numeric(df["power_kw"], errors="coerce")
    df["name"] = df["alias_name"].fillna(df["plant_name"]).fillna(df["plant_code"]).fillna(df["station_key"])
    return df

def _scrape_alias_by_key_map() -> Dict[str, str]:
    cfg = st.secrets.get("scraping", {}) or {}
    raw = dict(cfg.get("aliases_by_key", {}) or {})
    return {str(k).strip().lower(): str(v).strip() for k, v in raw.items()}

def _scrape_forced_shift_from_secrets() -> Optional[int]:
    cfg = st.secrets.get("scraping", {}) or {}
    v = cfg.get("rt_force_ts_shift_hours", None)
    try:
        return int(v) if v is not None else None
    except Exception:
        return None

def _scrape_pick_pvpp(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series([], dtype="object")
    inst = df.get("instance_key", pd.Series([""] * len(df))).astype(str).fillna("")
    pname = df.get("plant_name", pd.Series([""] * len(df))).astype(str).fillna("")
    pcode = df.get("plant_code", pd.Series([""] * len(df))).astype(str).fillna("")
    aname = df.get("alias_name", pd.Series([""] * len(df))).astype(str).fillna("")
    pvpp = aname.where(aname.str.len() > 0, other="")
    looks_pvpp = pcode.str.contains("_PVPP", case=False, na=False) | pcode.str.contains("PVPP", case=False, na=False)
    pvpp = pvpp.where(pvpp.str.len() > 0, other=pcode.where(looks_pvpp, other=""))
    amap = _scrape_alias_by_key_map()
    keys = (inst.str.strip() + "|" + pname.str.strip()).str.lower()
    mapped = keys.map(amap)
    pvpp = pvpp.where(pvpp.str.len() > 0, other=mapped.fillna(""))
    pvpp = pvpp.where(pvpp.str.len() > 0, other=pname.where(pname.str.len() > 0, other=pcode))
    return pvpp.fillna("")

def _infer_rt_shift_hours(df: pd.DataFrame) -> int:
    forced = RT_FORCE_TS_SHIFT_HOURS
    if forced is None:
        forced = _scrape_forced_shift_from_secrets()
    if forced is not None:
        return int(forced)
    if df is None or df.empty:
        return 0
    if "ts_utc" not in df.columns or "inserted_at" not in df.columns:
        return 0
    d = df.dropna(subset=["ts_utc", "inserted_at"]).copy()
    if d.empty:
        return 0
    d["ts_utc"] = pd.to_datetime(d["ts_utc"], utc=True, errors="coerce")
    d["inserted_at"] = pd.to_datetime(d["inserted_at"], utc=True, errors="coerce")
    d = d.dropna(subset=["ts_utc", "inserted_at"])
    d = d[d["ts_utc"].dt.year >= 2000]
    if len(d) < 50:
        return 0
    near = (d["ts_utc"] >= d["inserted_at"] - pd.Timedelta(hours=3)) & (d["ts_utc"] <= d["inserted_at"] + pd.Timedelta(minutes=45))
    d = d[near]
    if len(d) < 50:
        return 0
    delta_hours = (d["inserted_at"] - d["ts_utc"]).dt.total_seconds() / 3600.0
    med = float(delta_hours.median())
    if abs(med) < 0.8 or abs(med) > 3.2:
        return 0
    shift = int(round(med))
    if shift == 0 or abs(shift) > 3:
        return 0
    return shift

def _scrape_enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    for c in ["plant_code", "plant_name", "alias_name", "instance_key", "power_kw", "ts_utc", "inserted_at"]:
        if c not in d.columns:
            d[c] = None
    d["instance_key"] = d["instance_key"].fillna("unknown").astype(str)
    d["plant_code"] = d["plant_code"].fillna("").astype(str)
    d["plant_name"] = d["plant_name"].fillna("").astype(str)
    d["alias_name"] = d["alias_name"].fillna("").astype(str)
    d["ts_utc"] = pd.to_datetime(d["ts_utc"], utc=True, errors="coerce")
    d["inserted_at"] = pd.to_datetime(d["inserted_at"], utc=True, errors="coerce")
    d["power_kw"] = pd.to_numeric(d["power_kw"], errors="coerce")
    shift_h = _infer_rt_shift_hours(d)
    if shift_h != 0:
        d["ts_utc"] = d["ts_utc"] + pd.Timedelta(hours=shift_h)
    ts_eff = d["ts_utc"].copy()
    bad = ts_eff.isna() | (ts_eff.dt.year < 2000)
    ts_eff.loc[bad] = d.loc[bad, "inserted_at"]
    d["ts_eff_utc"] = ts_eff
    d["ts_local"] = d["ts_eff_utc"].dt.tz_convert(FS_TZ)
    d["plant_uid"] = (d["instance_key"].astype(str) + "|" + d["plant_code"].astype(str)).astype(str)
    d["PVPP"] = _scrape_pick_pvpp(d)
    d["name_series"] = d["PVPP"].where(d["PVPP"].astype(str).str.len() > 0, other=d["alias_name"])
    d["name_series"] = d["name_series"].where(d["name_series"].astype(str).str.len() > 0, other=d["plant_name"])
    d["name_series"] = d["name_series"].where(d["name_series"].astype(str).str.len() > 0, other=d["plant_code"])
    d = d.dropna(subset=["plant_uid", "ts_eff_utc"]).sort_values("ts_eff_utc", ascending=True).reset_index(drop=True)
    return d

def _sb_select_history_rt(hours_back: int = 48) -> pd.DataFrame:
    since = (_now_utc() - datetime.timedelta(hours=hours_back)).isoformat()
    select_cols = "ts_utc,inserted_at,plant_code,plant_name,alias_name,instance_key,power_kw"
    df = _sb_fetch_paged(
        table=SB_TABLE_SCRAPE,
        select_cols=select_cols,
        filter_col="inserted_at",
        since_iso=since,
        order_col="inserted_at",
        desc=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    return _scrape_enrich_df(df)

def _latest_per_plant_not_future_rt(df_rt: pd.DataFrame) -> pd.DataFrame:
    if df_rt is None or df_rt.empty:
        return pd.DataFrame()
    d = df_rt.copy()
    if "plant_uid" not in d.columns or "ts_eff_utc" not in d.columns:
        d = _scrape_enrich_df(d)
        if d.empty:
            return pd.DataFrame()
    now = _now_utc() + datetime.timedelta(minutes=FUTURE_TOL_MIN)
    d = d[d["ts_eff_utc"] <= now].copy()
    if d.empty:
        return pd.DataFrame()
    d = d.sort_values("ts_eff_utc", ascending=True)
    return d.groupby("plant_uid", as_index=False).tail(1).reset_index(drop=True)

def _render_main_metrics_table(df_live: pd.DataFrame):
    if df_live is None or df_live.empty:
        st.warning("Click Refresh.")
        return
    df2 = df_live.copy()
    df2["Putere_RT_kW"] = pd.to_numeric(df2["Putere_RT_kW"], errors="coerce")
    mask_excl = df2["Nume"].map(_is_excluded_name)
    sum_kw = float(df2.loc[~mask_excl, "Putere_RT_kW"].fillna(0).sum())
    total = len(df2)
    active_cnt = int(pd.to_numeric(df2["Active"], errors="coerce").fillna(0).astype(bool).sum())
    inactive_cnt = total - active_cnt
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total", total)
    m2.metric("Active", active_cnt)
    m3.metric("Inactive", inactive_cnt)
    m4.metric("Suma (kW)", f"{sum_kw:,.3f}".replace(",", " "))
    st.dataframe(df2[["Nume", "Active", "Putere_RT_kW"]], use_container_width=True, hide_index=True)

def _render_main_total_chart(hours_back: int = 24):
    st.subheader("Grafic TOTAL ")
    df_hist = _sb_load_main_last_hours(hours_back=hours_back)
    if df_hist is None or df_hist.empty:
        st.warning("Nu există snapshots.")
        return
    df_plot = df_hist.dropna(subset=["ts_local", "power_kw", "name"]).copy()
    df_plot["ts_local"] = pd.to_datetime(df_plot["ts_local"], errors="coerce").dt.tz_localize(None)
    wide = (
        df_plot.pivot_table(index="ts_local", columns="name", values="power_kw", aggfunc="last")
        .sort_index()
    )
    if wide.empty:
        st.warning("Nu pot construi graficul.")
        return
    cols_keep = [c for c in wide.columns if not _is_excluded_name(c)]
    wide2 = wide[cols_keep] if cols_keep else wide
    total = wide2.sum(axis=1, skipna=True).to_frame("TOTAL_kW")
    st.line_chart(total)

def _render_rt_metrics_like_main(df_rt: pd.DataFrame, title: str = "Scraping – Snapshot"):
    st.subheader(title)
    if df_rt is None or df_rt.empty:
        st.warning("Nu există date RT.")
        return
    if "plant_uid" not in df_rt.columns or "ts_eff_utc" not in df_rt.columns:
        df_rt = _scrape_enrich_df(df_rt)
        if df_rt.empty:
            st.warning("Eroare snapshot RT.")
            return

    latest = _latest_per_plant_not_future_rt(df_rt)
    if latest is None or latest.empty:
        st.warning("Nu găsesc valori valide.")
        return

    latest["power_kw"] = pd.to_numeric(latest["power_kw"], errors="coerce").fillna(0.0)

    active_mask = latest["power_kw"] > 0.0
    total = int(len(latest))
    active_cnt = int(active_mask.sum())
    inactive_cnt = total - active_cnt

    sum_kw = float(latest.loc[~latest["name_series"].map(_is_excluded_name), "power_kw"].fillna(0.0).sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total", total)
    m2.metric("Active", active_cnt)
    m3.metric("Inactive", inactive_cnt)
    m4.metric("Suma (kW)", f"{sum_kw:,.3f}".replace(",", " "))

    ts_snap_local = latest["ts_eff_utc"].max().astimezone(FS_TZ).strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"Snapshot: {ts_snap_local}")

    view = _agg_latest_for_table(latest)
    st.dataframe(view, use_container_width=True, hide_index=True)


def _render_rt_total_chart(df_rt: pd.DataFrame, title: str = "Scraping – Grafic TOTAL"):
    st.subheader(title)
    if df_rt is None or df_rt.empty:
        st.warning("Nu există date RT.")
        return

    wide = _wide_sum_by_name_series(df_rt)
    if wide is None or wide.empty:
        st.warning("Eroare date grafic.")
        return

    cols_keep = [c for c in wide.columns if not _is_excluded_name(c)]
    wide2 = wide[cols_keep] if cols_keep else wide

    total_series = wide2.sum(axis=1, skipna=True).to_frame("TOTAL_kW")
    st.line_chart(total_series)

def _agg_latest_for_table(latest: pd.DataFrame) -> pd.DataFrame:
    if latest is None or latest.empty:
        return pd.DataFrame(columns=["Nume", "Putere (kW)", "instance_key"])

    d = latest.copy()
    d["name_series"] = d["name_series"].fillna("").astype(str)
    d["instance_key"] = d["instance_key"].fillna("").astype(str)
    d["power_kw"] = pd.to_numeric(d["power_kw"], errors="coerce").fillna(0.0)

    def _join_instances(s: pd.Series) -> str:
        vals = [x for x in pd.unique(s.astype(str)) if x and x != "nan"]
        vals = sorted(vals)
        return ", ".join(vals)

    out = (
        d.groupby("name_series", as_index=False)
        .agg(power_kw=("power_kw", "sum"), instance_key=("instance_key", _join_instances))
        .rename(columns={"name_series": "Nume", "power_kw": "Putere (kW)"})
    )

    out["Putere (kW)"] = pd.to_numeric(out["Putere (kW)"], errors="coerce").fillna(0.0)
    out = out.sort_values(["Nume"], ascending=True).reset_index(drop=True)
    return out


def _wide_sum_by_name_series(df_rt: pd.DataFrame) -> pd.DataFrame:
    if df_rt is None or df_rt.empty:
        return pd.DataFrame()

    d = df_rt.copy()
    if "ts_eff_utc" not in d.columns or "plant_uid" not in d.columns or "name_series" not in d.columns:
        d = _scrape_enrich_df(d)
        if d.empty:
            return pd.DataFrame()

    now = _now_utc() + datetime.timedelta(minutes=FUTURE_TOL_MIN)
    d = d[d["ts_eff_utc"] <= now].copy()

    d["power_kw"] = pd.to_numeric(d["power_kw"], errors="coerce").fillna(0.0)
    d["ts_local_naive"] = d["ts_eff_utc"].dt.tz_convert(FS_TZ).dt.tz_localize(None)

    d = d.sort_values(["plant_uid", "ts_eff_utc"], ascending=[True, True])
    d = d.drop_duplicates(subset=["plant_uid", "ts_local_naive"], keep="last")

    grouped = (
        d.groupby(["ts_local_naive", "name_series"], as_index=False)["power_kw"]
        .sum()
    )

    wide = grouped.pivot(index="ts_local_naive", columns="name_series", values="power_kw").sort_index()
    return wide

def render_page():
    st.session_state.setdefault("fs_refresh_cooldown_until", 0.0)
    st.session_state.setdefault("fs_last_refresh", "-")
    st.session_state.setdefault("fs_df_all", None)
    st.session_state.setdefault("rt_hist_cache", None)
    st.session_state.setdefault("rt_hist_last_load", "-")
    st.title("Dashboard")
    st.header("Measurement APIs")
    try:
        title, token_ttl_sec, instances = _load_instances_from_secrets()
        st.caption(title)
    except Exception as e:
        st.error("Eroare secrets.toml.")
        st.exception(e)
        return
    c1, c2, _ = st.columns([1, 1, 6])
    refresh_main = c1.button("Refresh ", key="btn_refresh_main")
    show_chart_main = c2.button("Reload chart", key="btn_reload_chart_main")
    if refresh_main:
        now = time.time()
        cooldown_until = float(st.session_state.get("fs_refresh_cooldown_until", 0.0))
        if now < cooldown_until:
            st.warning(f"Așteaptă {int(cooldown_until - now)}s.")
        else:
            st.session_state["fs_refresh_cooldown_until"] = now + 30
            with st.spinner("Actualizare date..."):
                df_raw = _fetch_table_all_instances(instances, token_ttl_sec=token_ttl_sec)
                if df_raw is None or df_raw.empty:
                    st.session_state["fs_df_all"] = df_raw
                    st.session_state["fs_last_refresh"] = _now_local_str()
                else:
                    df_raw = df_raw.copy()
                    df_raw["plant_name"] = df_raw.get("Nume", "").astype(str).fillna("")
                    df_alias = _apply_aliases_inplace(df_raw)
                    df_raw["alias_name"] = df_alias["Nume"].astype(str).fillna("")
                    df_raw["Nume"] = df_raw["alias_name"]
                    st.session_state["fs_df_all"] = df_raw
                    st.session_state["fs_last_refresh"] = _now_local_str()
                    ts = _now_utc().replace(microsecond=0)
                    df_ins = df_raw.copy()
                    df_ins["ts_utc"] = ts
                    df_ins["instance_key"] = df_ins["Instanță"].astype(str).fillna("")
                    df_ins["plant_code"] = df_ins["Cod"].astype(str).fillna("")
                    df_ins["power_kw"] = pd.to_numeric(df_ins["Putere_RT_kW"], errors="coerce").fillna(0.0)
                    df_ins["plant_name"] = df_ins.get("plant_name", "").astype(str).fillna("")
                    df_ins["alias_name"] = df_ins.get("alias_name", "").astype(str).fillna("")
                    df_ins = df_ins[["ts_utc", "instance_key", "plant_code", "plant_name", "alias_name", "power_kw"]].copy()
                    _sb_upsert_snapshot_main(df_ins)
    st.info(f"Ultimul refresh: {st.session_state.get('fs_last_refresh', '-')}")
    _render_main_metrics_table(st.session_state.get("fs_df_all"))
    st.divider()
    st.header("Grafic ")
    _render_main_total_chart(hours_back=24)
    st.divider()
    st.header("Scraping ")
    c1, c2, c3 = st.columns([1, 1, 6])
    reload_rt = c1.button("Reload ", key="btn_reload_rt")
    save_csv_rt = c2.button("Save CSV", key="btn_save_rt")
    if reload_rt or st.session_state.get("rt_hist_cache") is None:
        st.session_state["rt_hist_cache"] = _sb_select_history_rt(hours_back=48)
        st.session_state["rt_hist_last_load"] = _now_local_str()
    st.caption(f"Ultima încărcare RT: {st.session_state.get('rt_hist_last_load', '-')}")
    df_rt = st.session_state.get("rt_hist_cache")
    _render_rt_metrics_like_main(df_rt, "Scraping")
    st.divider()
    _render_rt_total_chart(df_rt, "Scraping")
    if save_csv_rt and df_rt is not None and not df_rt.empty:
        out = df_rt.copy()
        if "ts_local" in out.columns:
            out["ts_local_str"] = pd.to_datetime(out["ts_local"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            out["ts_local_str"] = ""
        csv_hist = out[["ts_local_str", "name_series", "power_kw", "plant_code", "instance_key"]].to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_hist, file_name="fs_power_snapshots_rt_48h.csv", mime="text/csv")

if __name__ == "__main__":
    render_page()