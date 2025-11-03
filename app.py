import requests
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime, timezone

BASE = "https://d266k7wxhw6o23.cloudfront.net/"

st.set_page_config(page_title="DSOC WBGT", layout="wide")

# -----------------------------
# HTTP + caching helpers
# -----------------------------
@st.cache_data(ttl=15 * 60)
def fetch_json(url: str):
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    return r.json()

# -----------------------------
# Stations + variable discovery
# -----------------------------
@st.cache_data(ttl=15 * 60)
def load_station_catalog() -> pd.DataFrame:
    manifest = fetch_json(BASE + "metadata/manifest.json")
    stations_key = manifest["stations"]["key"]
    stations = fetch_json(BASE + stations_key)
    df = pd.DataFrame(stations)
    df.rename(columns={"relativeName": "name", "lat": "latitude", "lon": "longitude"}, inplace=True)
    df["establishedAt"] = pd.to_datetime(df["establishedAt"], errors="coerce")
    return df

@st.cache_data(ttl=24 * 60 * 60)
def discover_variable_codes():
    """
    Returns sets of candidate codes for dewpoint, air temp, and RH.
    Handles multiple possible shapes of the variables manifest.
    """
    manifest = fetch_json(BASE + "metadata/manifest.json")
    vars_key = manifest.get("variables", {}).get("key")

    dew_codes = {"td", "dewpoint", "dew_point"}  # safe defaults
    t_codes   = {"ta", "tair", "temp"}
    rh_codes  = {"rh"}

    if vars_key:
        raw = fetch_json(BASE + vars_key)

        # normalize to list[dict]
        items = []
        if isinstance(raw, list):
            for v in raw:
                items.append(v if isinstance(v, dict) else {"name": str(v)})
        elif isinstance(raw, dict):
            if isinstance(raw.get("variables"), list):
                items = raw["variables"]
            else:
                for k, v in raw.items():
                    if isinstance(v, dict):
                        items.append({"code": k, **v})
                    else:
                        items.append({"code": k, "name": str(v)})

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

# -----------------------------
# Latest observation helpers
# -----------------------------
def _choose_latest_key(manifest_json):
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

        dp = _first_present(rec, var_codes["dew"])
        if dp is not None:
            try:
                return (round(float(dp), 2), rec.get("time") or rec.get("timestamp"))
            except Exception:
                pass

        tc = _first_present(rec, var_codes["t"])
        rh = _first_present(rec, var_codes["rh"])
        dp = _dew_from_t_rh(tc, rh)
        return (dp, rec.get("time") or rec.get("timestamp"))
    except Exception:
        return (None, None)

@st.cache_data(ttl=5 * 60, show_spinner=True)
def attach_dewpoints(stations_df: pd.DataFrame) -> pd.DataFrame:
    out = stations_df.copy()
    out["dewpoint_C"] = [
        fetch_station_dewpoint(abbrev)[0] for abbrev in out["abbrev"].tolist()
    ]
    return out

# -----------------------------
# Sidebar / filters
# -----------------------------
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

with st.spinner("Fetching latest dew point values…"):
    filtered = attach_dewpoints(filtered)

county_df = filtered[filtered["county"] == selected_county].copy()

# -----------------------------
# Map drawing helper (with fallback)
# -----------------------------
def draw_scatter_map(data: pd.DataFrame, zoom: int, height: int, center=None, color_col=None):
    """
    Use px.scatter_map when available; fall back to px.scatter_mapbox.
    If color_col is None OR all-NaN → render neutral points (no legend).
    """
    use_color = color_col and data[color_col].notna().any()
    color_arg = color_col if use_color else None

    try:
        fig = px.scatter_map(
            data,
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
            color=color_arg,
            color_continuous_scale="Viridis",
            zoom=zoom,
            height=height,
            map_style="carto-positron",
            center=center,
        )
    except AttributeError:
        # Older Plotly → fall back (may show deprecation warnings)
        fig = px.scatter_mapbox(
            data,
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
            color=color_arg,
            color_continuous_scale="Viridis",
            zoom=zoom,
            height=height,
            mapbox_style="carto-positron",
            center=center,
        )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    if use_color:
        fig.update_layout(coloraxis_colorbar=dict(title="Dew Point (°C)"))
    return fig

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([2.2, 1.8], gap="large")

with left:
    st.markdown("### Commonwealth of Kentucky")
    if filtered.empty:
        st.info("No stations match the filters.")
    else:
        fig_state = draw_scatter_map(filtered, zoom=6, height=620, center=None, color_col="dewpoint_C")
        st.plotly_chart(fig_state, use_container_width=True)

with right:
    st.markdown(f"### {selected_county} County")
    if county_df.empty:
        st.info(f"No stations found for {selected_county} with current filters.")
    else:
        center_lat = float(county_df["latitude"].mean())
        center_lon = float(county_df["longitude"].mean())
        fig_county = draw_scatter_map(
            county_df, zoom=9, height=320, center={"lat": center_lat, "lon": center_lon}, color_col="dewpoint_C"
        )
        st.plotly_chart(fig_county, use_container_width=True)

    st.markdown(f"### Readout — {selected_county} stations")
    if county_df.empty:
        st.stop()

    # Table: safe formatting even when dewpoint is None
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

    def fmt_dp(x):
        return "" if pd.isna(x) else f"{x:.2f}"

    def fmt_date(x):
        try:
            return x.date().isoformat()
        except Exception:
            return ""

    st.dataframe(
        pretty.style.format({"Dew Point (°C)": fmt_dp, "Established": fmt_date}),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Dew point is pulled via per-station year manifests and the variables manifest; "
    "if a station lacks an explicit dew-point field, it’s computed from air temperature and RH."
)
