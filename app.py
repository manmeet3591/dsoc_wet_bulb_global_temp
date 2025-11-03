import json
import requests
import pandas as pd
import plotly.express as px
import streamlit as st

BASE = "https://d266k7wxhw6o23.cloudfront.net/"

st.set_page_config(page_title="DSOC WBGT", layout="wide")

# --------- Data helpers ---------
@st.cache_data(ttl=15 * 60)
def fetch_json(url: str):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

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

df = load_station_catalog()

# --------- Sidebar controls ---------
st.sidebar.title("Controls")

# County picker (defaults to Warren)
counties = sorted(df["county"].dropna().unique().tolist())
default_idx = counties.index("Warren") if "Warren" in counties else 0
selected_county = st.sidebar.selectbox("County for detail view", counties, index=default_idx)

# Optional filters
with st.sidebar.expander("Filter stations"):
    has_soil = st.checkbox("Has soil sensors", value=False)
    has_inversion = st.checkbox("Has inversion sensors", value=False)
    has_camera = st.checkbox("Has camera", value=False)

# Apply filters
filtered = df.copy()
if has_soil:
    filtered = filtered[filtered["hasSoil"] == 1]
if has_inversion:
    filtered = filtered[filtered["hasInversion"] == 1]
if has_camera:
    filtered = filtered[filtered["hasCamera"] == 1]

# County subset (for the zoomed panel + readout)
county_df = filtered[filtered["county"] == selected_county].copy()

# --------- Layout ---------
left, right = st.columns([2.2, 1.8], gap="large")

with left:
    st.markdown("### Commonwealth of Kentucky")
    # Statewide map
    if filtered.empty:
        st.info("No stations match the filters.")
    else:
        fig_state = px.scatter_mapbox(
            filtered,
            lat="latitude",
            lon="longitude",
            hover_name="name",
            hover_data={
                "abbrev": True,
                "county": True,
                "elevation": True,
                "timezone": True,
                "latitude": False,
                "longitude": False,
            },
            color="county",
            size=None,
            zoom=6,
            height=620,
        )
        # A neutral, no-token style
        fig_state.update_layout(mapbox_style="carto-positron", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_state, use_container_width=True)

with right:
    # Top: Warren (or chosen county) zoom map
    st.markdown(f"### {selected_county} County")

    if county_df.empty:
        st.info(f"No stations found for {selected_county} with current filters.")
    else:
        # Center map on the county’s stations
        center_lat = county_df["latitude"].mean()
        center_lon = county_df["longitude"].mean()

        fig_county = px.scatter_mapbox(
            county_df,
            lat="latitude",
            lon="longitude",
            hover_name="name",
            hover_data={"abbrev": True, "elevation": True, "timezone": True,
                        "latitude": False, "longitude": False},
            color_discrete_sequence=["#3b82f6"],
            zoom=9,
            height=320,
        )
        fig_county.update_layout(
            mapbox_style="carto-positron",
            mapbox_center={"lat": center_lat, "lon": center_lon},
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig_county, use_container_width=True)

    # Bottom: readout panel for the chosen county (easy place to add live variables)
    st.markdown(f"### Readout — {selected_county} stations")
    if county_df.empty:
        st.stop()

    display_cols = [
        "abbrev",
        "name",
        "elevation",
        "establishedAt",
        "hasSoil",
        "hasInversion",
        "hasCamera",
    ]
    # nicer headers
    pretty = county_df[display_cols].rename(
        columns={
            "abbrev": "ID",
            "name": "Station",
            "elevation": "Elev (ft)",
            "establishedAt": "Established",
            "hasSoil": "Soil",
            "hasInversion": "Inversion",
            "hasCamera": "Camera",
        }
    ).sort_values("Station")

    st.dataframe(
        pretty.style.format(
            {
                "Established": lambda t: t.date().isoformat() if pd.notnull(t) else "",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Tip: When you receive the variables doc, you can augment this panel with live values "
        "(e.g., temperature, RH, solar) and compute WBGT. The Mesonet email describes how to use "
        "the per-station **year manifest** (e.g., `data/FARM/2025/manifest.json`)."
    )

# --------- (Optional) WBGT hook ----------
# Placeholder function showing where you'd fetch latest observations and compute WBGT
# once you have variable abbreviations/units.
#
# def fetch_latest_for_station(abbrev: str) -> dict:
#     year = pd.Timestamp.now(tz="UTC").year
#     station_manifest_url = f"{BASE}data/{abbrev}/{year}/manifest.json"
#     m = fetch_json(station_manifest_url)
#     # parse m for latest chunk → download → compute metrics
#     return {}
