import time
import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from supabase import create_client
import fusion_api as fs  # <- modulul tău existent cu login / get_plants / kpi etc.

FS_TZ = ZoneInfo("Europe/Bucharest")
FS_SUM_EXCLUDE_NAME_CONTAINS = ["raal", "transavia","aldgate"]


def _now_local_str() -> str:
    return datetime.datetime.now(tz=FS_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _norm_name(s: str) -> str:
    return " ".join(str(s or "").strip().split())


def _norm_name_key(s: str) -> str:
    return _norm_name(s).lower()


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
        raise RuntimeError("Nu există instanțe în secrets.toml la [fusionsolar.instances.*].")

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
        raise RuntimeError("Instanțele există, dar lipsesc câmpuri obligatorii (username/system_code).")

    return title, token_ttl_sec, instances


def _decode_error_reason(msg: str) -> str:
    low = (msg or "").lower()
    if "failcode" in low and "407" in low:
        return "RATE LIMIT (407) – ACCESS_FREQUENCY_IS_TOO_HIGH"
    if "20400" in low or "user_or_value_invalid" in low:
        return "LOGIN INVALID (20400) – user/system_code greșite sau nu e cont Northbound"
    if "20056" in low and ("not authorized" in low or "not authorized by the owner" in low):
        return "FĂRĂ DREPTURI (20056) – contul API nu e autorizat"
    return msg


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
        raise RuntimeError("LOGIN RATE-LIMIT: așteaptă ~60s înainte de retry")
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
        raise RuntimeError("Lipsesc supabase.url / supabase.anon_key (și/sau service_role_key) în secrets.toml")
    return create_client(url, key)


def _sb_load_last_hours(hours_back: int = 24) -> pd.DataFrame:
    sb = _sb_client(service=False)
    since_utc = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours_back)).isoformat()

    res = (
        sb.table("fs_power_snapshots")
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


def _sb_upsert_snapshot_from_df(df_ins: pd.DataFrame) -> None:
    if df_ins is None or df_ins.empty:
        return

    sb = _sb_client(service=True)
    df2 = df_ins.copy()

    df2["ts_utc"] = pd.to_datetime(df2.get("ts_utc", datetime.datetime.now(datetime.timezone.utc)), utc=True, errors="coerce")
    df2["ts_utc"] = df2["ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    for col in ["instance_key", "plant_code", "plant_name", "alias_name"]:
        if col not in df2.columns:
            df2[col] = ""
        df2[col] = df2[col].astype(str).fillna("")

    if "power_kw" not in df2.columns:
        df2["power_kw"] = 0.0
    df2["power_kw"] = pd.to_numeric(df2["power_kw"], errors="coerce").fillna(0.0).astype(float)

    if "station_key" not in df2.columns:
        df2["station_key"] = (df2["instance_key"] + "|" + df2["plant_code"]).astype(str)
    df2["station_key"] = df2["station_key"].astype(str).fillna("")

    df2 = df2[df2["plant_code"].astype(str).str.len() > 0].copy()
    df2 = df2.drop_duplicates(subset=["ts_utc", "station_key"], keep="last")

    recs = df2[["ts_utc", "station_key", "instance_key", "plant_code", "plant_name", "alias_name", "power_kw"]].to_dict("records")
    if recs:
        sb.table("fs_power_snapshots").upsert(recs, on_conflict="ts_utc,station_key").execute()


def render_api_verification_page():
    title, token_ttl_sec, instances = _load_instances_from_secrets()

    st.session_state.setdefault("fs_df_all", None)
    st.session_state.setdefault("fs_last_refresh", "-")
    st.session_state.setdefault("fs_refresh_cooldown_until", 0.0)

    st.title(title)

    c1, c2, _ = st.columns([1, 1, 6])
    refresh = c1.button("Refresh")
    save_csv = c2.button("Save CSV")

    if refresh:
        now = time.time()
        cooldown_until = float(st.session_state.get("fs_refresh_cooldown_until", 0.0))
        if now < cooldown_until:
            st.warning(f"Refresh blocat (anti-spam). Mai așteaptă {int(cooldown_until - now)}s.")
        else:
            st.session_state["fs_refresh_cooldown_until"] = now + 30

            with st.spinner("Citesc puterile RT (FusionSolar) + salvez snapshot (Supabase)..."):
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

                    ts = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
                    df_ins = df_raw.copy()
                    df_ins["ts_utc"] = ts
                    df_ins["instance_key"] = df_ins["Instanță"].astype(str).fillna("")
                    df_ins["plant_code"] = df_ins["Cod"].astype(str).fillna("")
                    df_ins["power_kw"] = pd.to_numeric(df_ins["Putere_RT_kW"], errors="coerce").fillna(0.0)
                    df_ins["station_key"] = (df_ins["instance_key"] + "|" + df_ins["plant_code"]).astype(str)
                    df_ins = df_ins[["ts_utc", "station_key", "instance_key", "plant_code", "plant_name", "alias_name", "power_kw"]].copy()

                    _sb_upsert_snapshot_from_df(df_ins)

    df = st.session_state.get("fs_df_all")
    st.info(f"Ultimul refresh: {st.session_state.get('fs_last_refresh', '-')}")

    if df is None or df.empty:
        st.warning("Click refresh.")
    else:
        df_sum = df.copy()
        df_sum["Putere_RT_kW"] = pd.to_numeric(df_sum["Putere_RT_kW"], errors="coerce")
        mask_excl = df_sum["Nume"].astype(str).str.lower().str.contains("|".join(FS_SUM_EXCLUDE_NAME_CONTAINS), na=False)
        sum_kw = float(df_sum.loc[~mask_excl, "Putere_RT_kW"].fillna(0).sum())

        total = len(df)
        active_cnt = int(pd.to_numeric(df["Active"], errors="coerce").fillna(0).astype(bool).sum())
        inactive_cnt = total - active_cnt

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total", total)
        m2.metric("Active", active_cnt)
        m3.metric("Inactive", inactive_cnt)
        m4.metric("Suma (kW)", f"{sum_kw:,.3f}".replace(",", " "))

        df_view = df[["Nume", "Active", "Putere_RT_kW"]].copy()
        st.dataframe(
            df_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Active": st.column_config.CheckboxColumn("Active"),
                "Putere_RT_kW": st.column_config.NumberColumn("Putere RT (kW)", format="%.3f"),
            },
        )

    st.divider()
    st.title("Grafic ")
    df_hist = _sb_load_last_hours(hours_back=24)

    if df_hist is None or df_hist.empty:
        st.warning("Click refresh for snapshots")
        return

    df_plot = df_hist.dropna(subset=["ts_local", "power_kw", "name"]).copy()
    df_plot["ts_local"] = pd.to_datetime(df_plot["ts_local"], errors="coerce").dt.tz_localize(None)
    wide = (
        df_plot
        .pivot_table(index="ts_local", columns="name", values="power_kw", aggfunc="last")
        .sort_index()
    )

    if wide.empty:
        st.warning("Există date, dar nu pot construi seriile pentru grafic.")
        return

    # --- DOAR SUMA ---
    total = wide.sum(axis=1, skipna=True).to_frame("TOTAL_kW")
    pattern = "|".join(FS_SUM_EXCLUDE_NAME_CONTAINS)
    cols_keep = [c for c in wide.columns if pattern not in str(c).lower()]  # exclude după nume coloană
    wide2 = wide[cols_keep] if cols_keep else wide

    total = wide2.sum(axis=1, skipna=True).to_frame("TOTAL_kW")
    st.line_chart(total)

    if save_csv:
        csv_hist = df_hist[["ts_local", "name", "power_kw", "plant_code"]].to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_hist, file_name="fusionsolar_rt_snapshots_24h.csv", mime="text/csv")


if __name__ == "__main__":
    render_api_verification_page()
