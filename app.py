import math, io, html
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

import streamlit as st

st.set_page_config(page_title="LC LYNC LOS Feasibility", page_icon="📡", layout="wide")

# Insert CSS right here:
st.markdown("""
    <style>
    /* Hide top-right action buttons (GitHub link, settings, deploy) */
    [data-testid="stToolbar"] { display: none !important; }
    #MainMenu { display: none !important; }
    footer { display: none !important; }
    .stDeployButton { display: none !important; }
    </style>
""", unsafe_allow_html=True)
R = 6371000.0  # Earth radius in meters
...

# ==========================================
# GEOMETRY & ELEVATION HELPERS
# ==========================================
def hav(a, b, c, d):
    p1, p2 = map(math.radians, [a, c])
    dp = math.radians(c - a)
    dl = math.radians(d - b)
    x = math.sin(dp / 2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2)**2
    return 2 * R * math.atan2(math.sqrt(x), math.sqrt(1 - x))

def bearing(a, b, c, d):
    p1, p2 = map(math.radians, [a, c])
    dl = math.radians(d - b)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def point_at_dist(lat, lon, b, dist):
    p, l, br = map(math.radians, [lat, lon, b])
    q = dist / R
    la = math.asin(math.sin(p) * math.cos(q) + math.cos(p) * math.sin(q) * math.cos(br))
    lo = l + math.atan2(math.sin(br) * math.sin(q) * math.cos(p), math.cos(q) - math.sin(p) * math.sin(la))
    return math.degrees(la), (math.degrees(lo) + 540) % 360 - 180

@st.cache_data(ttl=3600)
def fetch_elevation_batch(coords_tuple):
    # Open-Meteo accepts comma-separated coordinate lists
    lats = ",".join(f"{c[0]:.6f}" for c in coords_tuple)
    lons = ",".join(f"{c[1]:.6f}" for c in coords_tuple)
    r = requests.get(
        "https://api.open-meteo.com/v1/elevation",
        params={"latitude": lats, "longitude": lons},
        timeout=45
    )
    r.raise_for_status()
    j = r.json()
    if isinstance(j, list):
        return [float(x["elevation"]) for x in j]
    return [float(x) for x in j["elevation"]]

def compute_profile(a, b, c, d, n=100):
    dist = hav(a, b, c, d)
    az_ab = bearing(a, b, c, d)
    az_ba = (az_ab + 180) % 360
    ds = np.linspace(0, dist, n)
    pts = [point_at_dist(a, b, az_ab, float(x)) for x in ds]
    
    # Query in batches of 100 points
    elevs = []
    for i in range(0, n, 100):
        elevs += fetch_elevation_batch(tuple(pts[i:i+100]))
    
    df = pd.DataFrame({
        "distance_m": ds,
        "latitude": [p[0] for p in pts],
        "longitude": [p[1] for p in pts],
        "terrain_elevation_m": elevs
    })
    df["terrain_elevation_m"] = df["terrain_elevation_m"].interpolate().bfill().ffill()
    return dist, az_ab, az_ba, df

def analyze_los(df, ha, hb, ga, gb, beam):
    D = float(df.distance_m.iloc[-1])
    x = df.distance_m.to_numpy()
    t = df.terrain_elevation_m.to_numpy()
    bulge = x * (D - x) / (2 * R)
    alt_a = ga + ha
    alt_b = gb + hb
    center = alt_a + (alt_b - alt_a) * x / D
    lower = center - beam / 2
    upper = center + beam / 2
    clear = lower - (t + bulge)
    
    test = clear.copy()
    test[[0, -1]] = np.inf
    crit_idx = int(np.argmin(test))
    
    obs = t + bulge + beam / 2
    f = x / D
    ra = (obs - f * alt_b) / (1 - f)
    ra[[0, -1]] = -np.inf
    ia = int(np.argmax(ra))
    
    rb = (obs - (1 - f) * alt_a) / f
    rb[[0, -1]] = -np.inf
    ib = int(np.argmax(rb))
    
    delta = obs - center
    delta[[0, -1]] = -np.inf
    de = max(0.0, float(np.max(delta)))
    
    out = df.copy()
    out["earth_curvature_m"] = bulge
    out["centerline_los_m"] = center
    out["beam_lower_edge_m"] = lower
    out["beam_upper_edge_m"] = upper
    out["beam_clearance_m"] = clear
    return out, crit_idx, max(0.0, float(ra[ia] - ga)), max(0.0, float(rb[ib] - gb)), ha + de, hb + de

# ==========================================
# KML GENERATOR
# ==========================================
def generate_kml(sites_data):
    """Generates standard Google Earth 3D KML for one or many links."""
    kml = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<kml xmlns="http://www.opengis.net/kml/2.2">',
           '<Document>',
           '<name>LC LYNC LOS Surveys</name>',
           '<Style id="clearLine"><LineStyle><color>ff00ff00</color><width>4</width></LineStyle></Style>',
           '<Style id="blockedLine"><LineStyle><color>ff0000ff</color><width>4</width></LineStyle></Style>']
    
    for s in sites_data:
        style = "#clearLine" if s["status"] == "CLEAR" else "#blockedLine"
        kml.append(f"""
        <Folder>
            <name>{html.escape(s['name_a'])} to {html.escape(s['name_b'])}</name>
            <Placemark>
                <name>{html.escape(s['name_a'])}</name>
                <Point><coordinates>{s['lon_a']},{s['lat_a']},{s.get('alt_a',0)}</coordinates></Point>
            </Placemark>
            <Placemark>
                <name>{html.escape(s['name_b'])}</name>
                <Point><coordinates>{s['lon_b']},{s['lat_b']},{s.get('alt_b',0)}</coordinates></Point>
            </Placemark>
            <Placemark>
                <name>Path: {s['status']}</name>
                <styleUrl>{style}</styleUrl>
                <LineString>
                    <extrude>1</extrude>
                    <altitudeMode>absolute</altitudeMode>
                    <coordinates>
                        {s['lon_a']},{s['lat_a']},{s.get('alt_a',0)}
                        {s['lon_b']},{s['lat_b']},{s.get('alt_b',0)}
                    </coordinates>
                </LineString>
            </Placemark>
        </Folder>
        """)
    kml.extend(['</Document>', '</kml>'])
    return "".join(kml)

# ==========================================
# EXCEL IMPORT NORMALIZATION
# ==========================================
def normalize_excel(df):
    aliases = {
        "site_a_name": ["site a name", "site a", "site_a_name", "site_a"],
        "site_a_lat": ["site a latitude", "site a lat", "lat a", "latitude a", "site_a_lat"],
        "site_a_lon": ["site a longitude", "site a long", "lon a", "longitude a", "site_a_lon"],
        "site_b_name": ["site b name", "site b", "site_b_name", "site_b"],
        "site_b_lat": ["site b latitude", "site b lat", "lat b", "latitude b", "site_b_lat"],
        "site_b_lon": ["site b longitude", "site b long", "lon b", "longitude b", "site_b_lon"],
        "height_a": ["height a", "site a height", "height_a"],
        "height_b": ["height b", "site b height", "height_b"],
        "power_mode": ["power mode", "device power mode", "power_mode"],
        "power_cable_length_m": ["power cable length", "power cable length m", "power_cable_length_m"],
        "data_output": ["data output", "data output port", "data_output"],
        "data_cable_length_m": ["data cable length", "data cable length m", "data_cable_length_m"],
        "beam_diameter_m": ["beam diameter", "beam diameter m", "beam_diameter_m"],
    }
    cols = {str(c).strip().lower().replace("_", " "): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for target, names in aliases.items():
        found = None
        for n in names:
            key = n.lower().replace("_", " ")
            if key in cols:
                found = cols[key]
                break
        out[target] = df[found] if found is not None else np.nan
    return out

# ==========================================
# SIDEBAR NAVIGATION & INPUTS
# ==========================================
st.sidebar.title("LC LYNC LOS FEASIBILITY")
mode = st.sidebar.radio("Analysis Mode", ["Single Link", "Multiple Links"])

active_survey = None

if mode == "Single Link":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Site A")
    name_a = st.sidebar.text_input("Name", "Site A-001", key="s_na")
    lat_a = st.sidebar.number_input("Lat", min_value=-90.0, max_value=90.0, value=19.1533240, format="%.7f", key="s_la")
    lon_a = st.sidebar.number_input("Long", min_value=-180.0, max_value=180.0, value=72.8509670, format="%.7f", key="s_loa")
    ha = st.sidebar.number_input("Height A (m AGL)", 0.0, 100.0, 10.0, step=0.5, key="s_ha")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Site B")
    name_b = st.sidebar.text_input("Name", "Site B-001", key="s_nb")
    lat_b = st.sidebar.number_input("Lat", min_value=-90.0, max_value=90.0, value=19.1510150, format="%.7f", key="s_lb")
    lon_b = st.sidebar.number_input("Long", min_value=-180.0, max_value=180.0, value=72.8500800, format="%.7f", key="s_lob")
    hb = st.sidebar.number_input("Height B (m AGL)", 0.0, 100.0, 10.0, step=0.5, key="s_hb")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Power & Data")
    power_mode = st.sidebar.selectbox("Power Mode", ["48V DC", "AC PoE", "220V AC"], key="s_pm")
    power_len = st.sidebar.number_input("Power Cable Length (m)", 0.0, 500.0, 20.0, key="s_pcl")
    data_port = st.sidebar.selectbox("Data Output", ["ETH", "SMF (Single Mode)", "MMF (Multimode)"], key="s_do")
    data_len = st.sidebar.number_input("Data Cable Length (m)", 0.0, 500.0, 30.0, key="s_dcl")
    beam_d = st.sidebar.slider("Beam Diameter (m)", 1.0, 5.0, 3.0, 0.5, key="s_beam")

    if st.sidebar.button("ANALYSE LOS", type="primary", use_container_width=True):
        with st.spinner("Calculating single link profile..."):
            D, az_ab, az_ba, prof_df = compute_profile(lat_a, lon_a, lat_b, lon_b, n=120)
            ga = float(prof_df.terrain_elevation_m.iloc[0])
            gb = float(prof_df.terrain_elevation_m.iloc[-1])
            calc_df, ci, reqa, reqb, eqa, eqb = analyze_los(prof_df, ha, hb, ga, gb, beam_d)
            crit = calc_df.iloc[ci]
            status = "CLEAR" if crit.beam_clearance_m >= 0 else "BLOCKED"
            
            st.session_state["single_result"] = {
                "name_a": name_a, "lat_a": lat_a, "lon_a": lon_a, "ha": ha, "ga": ga, "alt_a": ga + ha,
                "name_b": name_b, "lat_b": lat_b, "lon_b": lon_b, "hb": hb, "gb": gb, "alt_b": gb + hb,
                "power_mode": power_mode, "power_len": power_len, "data_port": data_port, "data_len": data_len,
                "beam": beam_d, "dist_m": D, "az_ab": az_ab, "az_ba": az_ba,
                "status": status, "clearance": crit.beam_clearance_m,
                "reqa": reqa, "reqb": reqb, "eqa": eqa, "eqb": eqb,
                "crit_dist": crit.distance_m, "crit_lat": crit.latitude, "crit_lon": crit.longitude,
                "profile_df": calc_df
            }

    if "single_result" in st.session_state:
        active_survey = st.session_state["single_result"]

else:  # Multiple Links Mode
    st.sidebar.markdown("---")
    st.sidebar.subheader("Multiple Links")
    up_file = st.sidebar.file_uploader("Upload Excel", type=["xlsx", "xls"])
    
    # Template download helper
    template_buf = io.BytesIO()
    with pd.ExcelWriter(template_buf, engine="openpyxl") as w:
        pd.DataFrame([{
            "Site A Name": "Site 1", "Site A Latitude": 19.153324, "Site A Longitude": 72.850967, "Height A": 10.0,
            "Site B Name": "Site 2", "Site B Latitude": 19.151015, "Site B Longitude": 72.850080, "Height B": 10.0,
            "Power Mode": "48V DC", "Power Cable Length (m)": 20, "Data Output": "ETH", "Data Cable Length (m)": 30,
            "Beam Diameter (m)": 3.0
        }]).to_excel(w, index=False)
    st.sidebar.download_button("Download Template", template_buf.getvalue(), "LOS_Input_Template.xlsx", use_container_width=True)

    if up_file:
        if st.sidebar.button("ANALYSE ALL", type="primary", use_container_width=True):
            raw = pd.read_excel(up_file)
            inp = normalize_excel(raw)
            results_dict = {}
            summary_records = []
            
            prog = st.sidebar.progress(0.0)
            total = len(inp)
            for idx, row in inp.iterrows():
                try:
                    na = str(row.site_a_name) if pd.notna(row.site_a_name) else f"Site A-{idx+1}"
                    nb = str(row.site_b_name) if pd.notna(row.site_b_name) else f"Site B-{idx+1}"
                    la, loa = float(row.site_a_lat), float(row.site_a_lon)
                    lb, lob = float(row.site_b_lat), float(row.site_b_lon)
                    h_a = float(row.height_a) if pd.notna(row.height_a) else 10.0
                    h_b = float(row.height_b) if pd.notna(row.height_b) else 10.0
                    bm = float(row.beam_diameter_m) if pd.notna(row.beam_diameter_m) else 3.0
                    pm = str(row.power_mode) if pd.notna(row.power_mode) else "48V DC"
                    plen = float(row.power_cable_length_m) if pd.notna(row.power_cable_length_m) else 0.0
                    dout = str(row.data_output) if pd.notna(row.data_output) else "ETH"
                    dlen = float(row.data_cable_length_m) if pd.notna(row.data_cable_length_m) else 0.0

                    D, az_ab, az_ba, prof_df = compute_profile(la, loa, lb, lob, n=100)
                    ga = float(prof_df.terrain_elevation_m.iloc[0])
                    gb = float(prof_df.terrain_elevation_m.iloc[-1])
                    calc_df, ci, reqa, reqb, eqa, eqb = analyze_los(prof_df, h_a, h_b, ga, gb, bm)
                    crit = calc_df.iloc[ci]
                    status = "CLEAR" if crit.beam_clearance_m >= 0 else "BLOCKED"

                    key_label = f"{idx+1}: {na} → {nb}"
                    rec = {
                        "Survey ID": idx + 1, "key": key_label, "name_a": na, "lat_a": la, "lon_a": loa, "ha": h_a, "ga": ga, "alt_a": ga + h_a,
                        "name_b": nb, "lat_b": lb, "lon_b": lob, "hb": h_b, "gb": gb, "alt_b": gb + h_b,
                        "power_mode": pm, "power_len": plen, "data_port": dout, "data_len": dlen, "beam": bm,
                        "dist_m": D, "az_ab": az_ab, "az_ba": az_ba, "status": status, "clearance": crit.beam_clearance_m,
                        "reqa": reqa, "reqb": reqb, "eqa": eqa, "eqb": eqb,
                        "crit_dist": crit.distance_m, "crit_lat": crit.latitude, "crit_lon": crit.longitude,
                        "profile_df": calc_df
                    }
                    results_dict[key_label] = rec
                    summary_records.append(rec)
                except Exception as ex:
                    st.sidebar.error(f"Row {idx+1} Error: {ex}")
                prog.progress((idx + 1) / total)

            st.session_state["batch_results"] = results_dict
            st.session_state["summary_records"] = summary_records
            st.sidebar.success(f"Processed {len(summary_records)} links successfully!")

    if "batch_results" in st.session_state and st.session_state["batch_results"]:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Select Survey")
        keys = list(st.session_state["batch_results"].keys())
        selected_key = st.sidebar.selectbox("▼ Choose Link", keys)
        active_survey = st.session_state["batch_results"][selected_key]
        
        # Batch and Selected Download Buttons
        st.sidebar.markdown("---")
        # 1. Download Selected KML
        sel_kml = generate_kml([active_survey])
        st.sidebar.download_button(
            "Download Selected KML", sel_kml, 
            file_name=f"{active_survey['name_a']}_{active_survey['name_b']}.kml", mime="application/vnd.google-earth.kml+xml",
            use_container_width=True
        )
        
        # 2. Download All KML
        all_kml = generate_kml(list(st.session_state["batch_results"].values()))
        st.sidebar.download_button(
            "Download All KML", all_kml, 
            file_name="All_Survey_Links.kml", mime="application/vnd.google-earth.kml+xml",
            use_container_width=True
        )
        
        # 3. Download Full Excel Report
        rep_buf = io.BytesIO()
        with pd.ExcelWriter(rep_buf, engine="openpyxl") as w:
            rep_df = pd.DataFrame([{k: v for k, v in r.items() if k != "profile_df"} for r in st.session_state["summary_records"]])
            rep_df.to_excel(w, index=False, sheet_name="LOS Summary Report")
        st.sidebar.download_button(
            "Download Excel Report", rep_buf.getvalue(),
            file_name="LC_LYNC_LOS_Report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# ==========================================
# MAIN SCREEN DISPLAY
# ==========================================
if active_survey:
    s = active_survey
    st.header("LINK SUMMARY")
    st.markdown("---")
    
    # Distance & Azimuths
    m1, m2, m3 = st.columns(3)
    m1.metric("Distance", f"{s['dist_m']/1000.0:.3f} km ({s['dist_m']:.1f} m)")
    m2.metric("Azimuth A → B", f"{s['az_ab']:.2f}°")
    m3.metric("Azimuth B → A", f"{s['az_ba']:.2f}°")

    st.write("")
    c_a, c_b = st.columns(2)
    with c_a:
        st.subheader("SITE A")
        st.markdown(f"**Point:** `{s['name_a']}`")
        st.write(f"**Lat:** `{s['lat_a']:.7f}`")
        st.write(f"**Long:** `{s['lon_a']:.7f}`")
        st.write(f"**Elevation:** `{s['ga']:.1f} m` | **Device Height:** `{s['ha']:.1f} m AGL`")
    with c_b:
        st.subheader("SITE B")
        st.markdown(f"**Point:** `{s['name_b']}`")
        st.write(f"**Lat:** `{s['lat_b']:.7f}`")
        st.write(f"**Long:** `{s['lon_b']:.7f}`")
        st.write(f"**Elevation:** `{s['gb']:.1f} m` | **Device Height:** `{s['hb']:.1f} m AGL`")

    st.write("")
    if s["status"] == "CLEAR":
        st.success(f"### 🟢 LOS FEASIBILITY: CLEAR (Margin: {s['clearance']:.2f} m)")
    else:
        st.error(f"### 🔴 LOS FEASIBILITY: BLOCKED (Obstruction by {abs(s['clearance']):.2f} m)")

    # Suggested Height Adjustments
    st.markdown("**Mounting Height Solutions to Clear Optical Beam:**")
    h1, h2, h3 = st.columns(3)
    h1.info(f"**Raise Site A Only:** {s['reqa']:.1f} m AGL")
    h2.info(f"**Raise Site B Only:** {s['reqb']:.1f} m AGL")
    h3.info(f"**Equal Raise:** {s['eqa']:.1f} m / {s['eqb']:.1f} m AGL")

    # SECTION: LOS / TERRAIN PROFILE
    st.markdown("---")
    st.header("LOS / TERRAIN PROFILE")
    fig = go.Figure()
    pdf = s["profile_df"]
    fig.add_trace(go.Scatter(x=pdf.distance_m, y=pdf.terrain_elevation_m, mode="lines", name="Terrain Elevation", line=dict(color="#8B5A2B", width=2)))
    fig.add_trace(go.Scatter(x=pdf.distance_m, y=pdf.centerline_los_m, mode="lines", name="Beam Centerline", line=dict(color="#1f77b4", dash="dash")))
    fig.add_trace(go.Scatter(x=pdf.distance_m, y=pdf.beam_lower_edge_m, mode="lines", name="Lower Edge", line=dict(color="#aec7e8", width=1)))
    fig.add_trace(go.Scatter(x=pdf.distance_m, y=pdf.beam_upper_edge_m, mode="lines", name="Upper Edge", line=dict(color="#aec7e8", width=1), fill="tonexty", fillcolor="rgba(31, 119, 180, 0.1)"))
    fig.add_trace(go.Scatter(x=[s["crit_dist"]], y=[pdf.loc[pdf.distance_m == s["crit_dist"], "terrain_elevation_m"].iloc[0]], mode="markers", name="Critical Point", marker=dict(size=12, color="red", symbol="x")))
    fig.update_layout(height=450, xaxis_title="Distance from Site A (m)", yaxis_title="Elevation (m)", hovermode="x unified", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    # SECTION: VISUAL INSPECTION
    st.markdown("---")
    st.header("VISUAL INSPECTION")
    center_lat = (s["lat_a"] + s["lat_b"]) / 2
    center_lon = (s["lon_a"] + s["lon_b"]) / 2
    folium_map = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles=None)
    folium.TileLayer("OpenStreetMap", name="Street Map").add_to(folium_map)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite"
    ).add_to(folium_map)
    
    path_color = "green" if s["status"] == "CLEAR" else "red"
    folium.Marker([s["lat_a"], s["lon_a"]], tooltip=s["name_a"], icon=folium.Icon(color="blue", icon="tower")).add_to(folium_map)
    folium.Marker([s["lat_b"], s["lon_b"]], tooltip=s["name_b"], icon=folium.Icon(color="darkblue", icon="tower")).add_to(folium_map)
    folium.PolyLine([[s["lat_a"], s["lon_a"]], [s["lat_b"], s["lon_b"]]], color=path_color, weight=4, opacity=0.8).add_to(folium_map)
    folium.CircleMarker([s["crit_lat"], s["crit_lon"]], radius=6, color="red", fill=True, tooltip="Worst Clearance Point").add_to(folium_map)
    folium.LayerControl().add_to(folium_map)
    st_folium(folium_map, use_container_width=True, height=500, key=f"map_{s.get('key', 'single')}")

    # SECTION: REPORT / EXPORT
    st.markdown("---")
    st.header("REPORT")
    single_rep_df = pd.DataFrame([{k: v for k, v in s.items() if k != "profile_df"}])
    st.dataframe(single_rep_df.T.rename(columns={0: "Survey Parameter"}), use_container_width=True)

    out_buf = io.BytesIO()
    with pd.ExcelWriter(out_buf, engine="openpyxl") as w:
        single_rep_df.to_excel(w, index=False, sheet_name="Survey Detail")
    
    r1, r2 = st.columns(2)
    with r1:
        st.download_button("Download Excel Report", out_buf.getvalue(), file_name=f"{s['name_a']}_{s['name_b']}_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    with r2:
        kml_out = generate_kml([s])
        st.download_button("Download KML", kml_out, file_name=f"{s['name_a']}_{s['name_b']}.kml", mime="application/vnd.google-earth.kml+xml", use_container_width=True)

else:
    st.info("👈 Select an Analysis Mode in the sidebar and run the calculation to view the LOS summary.")
