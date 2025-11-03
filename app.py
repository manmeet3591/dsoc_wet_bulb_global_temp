import json
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime, timezone

BASE = "https://d266k7wxhw6o23.cloudfront.net/"

st.set_page_config(page_title="DSOC WBGT", layout="wide")

# ======================
# HTTP + caching helpers
# ======================
@st.cache_data(ttl=15 * 60)
def fetch_json(url: str):
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    return r.json()

# =============================
# Discover stations + variables
# =============================
@st.cache_data(ttl=15 * 60)
def load_station_catalog() -> pd.DataFrame:
    # 1) Manifest → find current stations file (stable URL is the manifest)
    manifest = fetch_json(BASE + "metadata/manifest.json")
    stations_key = manifest["stations"]["key"]  # e.g. metadata/stations_<hash>.json
    stations = fetch_json(BASE + stations_key)

    df = pd.DataFrame(stations)
    # Tidy up types/labels
    df.rename(
        columns={
            "relativeName": "name",
            "lat": "latitude",
            "lon": "longitude",
        },
        inplace=True,
    )
    df["establishedAt"] = pd.to_datetime(df["establishedAt"], errors="coerce")
    return df

@st.cache_data(ttl=24 * 60 * 60)
def discover_variable_codes():
    """Return likely variable codes for dew point, air temp, RH. Robust to different manifest shapes."""
    manifest = fetch_json(BASE + "metadata/manifest.json")
    vars_key = manifest.get("variables", {}).get("key")

    dew_codes = {"td", "dewpoint", "dew_point"}   # safe defaults
    t_codes   = {"ta", "tair", "temp"}
    rh_codes  = {"rh"}

    if vars_key:
        raw = fetch_json(BASE + vars_key)

        # Normalize to a list of dicts
        items = []
        if isinstance(raw, dict):
            if isinstance(raw.get("variables"), list):
                items = raw["variables"]
            else:
                for k, v in raw.items():
                    if isinstance(v, dict):
                        items.append({"code": k, **v})
                    else:
                        items.append({"code": k, "name": str(v)})
        elif isinstance(raw, list):
            for v in raw:
                items.append(v if isinstance(v, dict) else {"name": str(v)})

        for item in items:
            text = " ".join(
                str(item.get(k, "")).lower()
                for k in ("name", "description", "abbr", "abbrev", "variable", "units")
            )
            code = str(item.get("abbrev") or item.get("abbr") or item.get("code") or "").lower()

            if ("dew" in text and "point" in text) or code in {"td", "dewpoint", "dew_point"}:
                if code:
                    dew_codes.add(code)
            if (("temperature" in text and ("air" in text or "dry" in text)) or code in {"ta", "tair", "temp"}):
                if code:
                    t_codes.add(code)
            if (("relative" in text and "humidity" in text) or code == "rh"):
                if code:
                    rh_codes.add(code)

    return {"dew": sorted(dew_codes), "t": sorted(t_codes), "rh": sorted(rh_codes)}

df = load_station_catalog()
var_codes = discover_variable_codes()

# ===================================
# Latest observation & dewpoint logic
# ===================================
def _choose_latest_key(manifest_json):
    """
    Year manifest can be a list of chunks with 'key' fields or a dict with 'key'.
    Prefer entries that look realtime and the newest timestamp if present.
    """
    if isinstance(manifest_json, list) and manifest_json:
        def score(item):
            k = str(item.get("key", "")).lower()
            s = 0
            if "real" in k or "rt" in k: s += 3
            if "5" in k and "min" in k: s += 2
            if k.endswith(".json"): s += 1
            ts = item.get("timestamp") or item.get("time") or item.get("updated")
            try:
                ts_dt = pd.to_datetime(ts, utc=True)
            except Exception:
                ts_dt = pd.Timestamp(0, tz="UTC")
            return (s, ts_dt)
        return max(manifest_json, key=score).get("key")
    if isinstance(manifest_json, dict) and "key" in manifest_json:
        return manifest_json["key"]
    return None

def _extract_latest_record(payload):
    """
    Try several shapes: list[dict] (timeseries), dict with 'data'/'records'/'observations'/'obs'.
    Return most recent record.
    """
    if isinstance(payload, list) and payload:
        return payload[-1] if isinstance(payload[-1], dict) else None
    if isinstance(payload, dict):
        for k in ("data", "records", "observations", "obs"):
            v = payload.get(k)
            if isinstance(v, list) and v and isinstance(v[-1], dict):
                return v[-1]
    return None

def _first_present(d: dict, keys):
    for k in keys:
        if k in d and pd.notna(d[k]):
            return d[k]
    return None

def _dew_from_t_rh(tc, rh):
    """Magnus formula: inputs °C and % → returns dew point °C."""
    if tc is None or rh is None:
        return None
    try:
        tc = float(tc)
        rh = float(rh)
        if rh <= 0 or rh > 100:
            return None
        a, b = 17.625, 243.04
        gamma = (a * tc) / (b + tc) + np.log(rh / 100.0)
        td = (b * gamma) / (a - gamma)
        return round(float(td), 2)
    except Exception:
        return None

@st.cache_data(ttl=5 * 60, show_spinner=False)
def fetch_station_dewpoint(abbrev: str):
    """Return (dewpoint_c, timestamp_iso) or (None, None)."""
    try:
        year = datetime.now(timezone.utc).year
        m_url = f"{BASE}data/{abbrev}/{year}/manifest.json"
        m = fetch_json(m_url)
        latest_key = _choose_latest_key(m)
        if not latest_key:
            return (None, None)
        data_url = BASE + latest_key
        payload = fetch_json(data_url)
        rec = _extract_latest_record(payload)
        if not isinstance(rec, dict):
            return (None, None)

        # Direct dew point if present
        dp = _first_present(rec, var_codes["dew"])
        if dp is not None:
            try:
                return (round(float(dp), 2), rec.get("time") or rec.get("timestamp"))
            except Exception:
                pass

        # Compute from T (°C) and RH (%)
        tc = _first_present(rec, var_codes["t"])
        rh = _first_present(rec, var_codes["rh"])
        dp = _dew_from_t_rh(tc, rh)
        return (dp, rec.get("time") or rec.get("timestamp"))
    except Exception:
        return (None, None)

@st.cache_data(ttl=5 * 60, show_spinner=True)
def attach_dewpoints(stations_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in stations_df.iterrows():
        dp, ts = fetch_station_dewpoint(r["abbrev"])
        rows.append(dp)
    out = stations_df.copy()
    out["dewpoint_C"] = rows
    return out

# ===================
# Sidebar / filtering
# ===================
st.sidebar.title("Controls")

counties = sorted(df["county"].dropna().unique().tolist())
default_idx = counties.index("Warren") if "Warren" in counties else 0
selected_county = st.sidebar.selectbox("County for detail view", counties, index=default_idx)

with st.sidebar.expander("Filter stations"):
    has_soil = st.checkbox("Has soil sensors", value=False)
    has_inversion = st.checkbox("Has inversion sensors", value=False)
    has_camera = st.checkbox("Has camera", value=False)

filtered = df.copy()
if has_soil:
    filtered = filtered[filtered["hasSoil"] == 1]
if has_inversion:
    filtered = filtered[filtered["hasInversion"] == 1]
if has_camera:
    filtered = filtered[filtered["hasCamera"] == 1]

# Fetch dew points (adds dewpoint_C column)
with st.spinner("Fetching latest dew point values…"):
    filtered = attach_dewpoints(filtered)

county_df = filtered[filtered["county"] == selected_county].copy()

# ==========
# Main layout
# ==========
left, right = st.columns([2.2, 1.8], gap="large")

with left:
    st.markdown("### Commonwealth of Kentucky")
    if filtered.empty:
        st.info("No stations match the filters.")
    else:
        # NOTE: We do NOT color by county → no big county legend
        # Use dew point as continuous color (set color=None to remove legend entirely)
        fig_state = px.scatter_map(
            filtered,
            lat="latitude",
            lon="longitude",
            hover_name="name",
            hover_data={
                "abbrev": True,
                "county": True,
                "dewpoint_C": True,
                "timezone": True,
                "latitude": False,
                "longitude": False,
            },
            color="dewpoint_C",
            color_continuous_scale="Viridis",
            zoom=6,
            height=620,
            map_style="carto-positron",
        )
        fig_state.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_colorbar=dict(title="Dew Point (°C)"),
        )
        st.plotly_chart(fig_state, use_container_width=True)

with right:
    st.markdown(f"### {selected_county} County")
    if county_df.empty:
        st.info(f"No stations found for {selected_county} with current filters.")
    else:
        center_lat = county_df["latitude"].mean()
        center_lon = county_df["longitude"].mean()

        fig_county = px.scatter_map(
            county_df,
            lat="latitude",
            lon="longitude",
            hover_name="name",
            hover_data={
                "abbrev": True,
                "dewpoint_C": True,
                "timezone": True,
                "latitude": False,
                "longitude": False,
            },
            color="dewpoint_C",
            color_continuous_scale="Viridis",
            zoom=9,
            height=320,
            map_style="carto-positron",
            center={"lat": center_lat, "lon": center_lon},
        )
        fig_county.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_colorbar=dict(title="Dew Point (°C)"),
        )
        st.plotly_chart(fig_county, use_container_width=True)

    st.markdown(f"### Readout — {selected_county} stations")
    if county_df.empty:
        st.stop()

    display_cols = [
        "abbrev",
        "name",
        "dewpoint_C",
        "establishedAt",
        "hasSoil",
        "hasInversion",
        "hasCamera",
    ]
    pretty = (
        county_df[display_cols]
        .rename(
            columns={
                "abbrev": "ID",
                "name": "Station",
                "dewpoint_C": "Dew Point (°C)",
                "establishedAt": "Established",
                "hasSoil": "Soil",
                "hasInversion": "Inversion",
                "hasCamera": "Camera",
            }
        )
        .sort_values("Station")
    )

    st.dataframe(
        pretty.style.format(
            {
                "Dew Point (°C)": "{:.2f}",
                "Established": lambda t: t.date().isoformat() if pd.notnull(t) else "",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Dew point is pulled via per-station year manifests and the variables manifest; "
    "if a station lacks an explicit dew-point field, it’s computed from air temperature and RH."
)
