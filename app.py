import math
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Velmenni LC LYNC LOS V2", page_icon="📡", layout="wide")

EARTH_RADIUS_M = 6_371_000.0

def haversine_distance(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def initial_bearing(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2-lon1)
    x = math.sin(dl)*math.cos(p2)
    y = math.cos(p1)*math.sin(p2) - math.sin(p1)*math.cos(p2)*math.cos(dl)
    return (math.degrees(math.atan2(x,y))+360)%360

def destination_point(lat, lon, bearing_deg, distance_m):
    p, l, b = map(math.radians, [lat, lon, bearing_deg])
    a = distance_m/EARTH_RADIUS_M
    np_lat = math.asin(math.sin(p)*math.cos(a)+math.cos(p)*math.sin(a)*math.cos(b))
    np_lon = l + math.atan2(math.sin(b)*math.sin(a)*math.cos(p),
                            math.cos(a)-math.sin(p)*math.sin(np_lat))
    np_lon = (math.degrees(np_lon)+540)%360-180
    return math.degrees(np_lat), np_lon

@st.cache_data(ttl=3600, show_spinner=False)
def get_elevations(coords):
    lats = ",".join(f"{p[0]:.7f}" for p in coords)
    lons = ",".join(f"{p[1]:.7f}" for p in coords)
    r = requests.get("https://api.open-meteo.com/v1/elevation",
                     params={"latitude": lats, "longitude": lons}, timeout=60)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list):
        return [float(x["elevation"]) for x in data]
    elev = data.get("elevation")
    if isinstance(elev, list):
        return [float(x) for x in elev]
    return [float(elev)] * len(coords)

def terrain_profile(lat1, lon1, lat2, lon2, distance_m, samples):
    bearing = initial_bearing(lat1, lon1, lat2, lon2)
    distances = np.linspace(0, distance_m, samples)
    coords = [destination_point(lat1, lon1, bearing, float(d)) for d in distances]
    elevations = []
    for i in range(0, len(coords), 100):
        elevations.extend(get_elevations(coords[i:i+100]))
    return pd.DataFrame({
        "distance_m": distances,
        "latitude": [c[0] for c in coords],
        "longitude": [c[1] for c in coords],
        "terrain_elevation_m": elevations
    })

def earth_bulge(x, D):
    return x*(D-x)/(2*EARTH_RADIUS_M)

def solve_required_a(df, ground_a, abs_b, beam_d):
    D = float(df.distance_m.iloc[-1])
    x = df.distance_m.to_numpy()
    obstacle = df.terrain_elevation_m.to_numpy() + earth_bulge(x,D) + beam_d/2
    frac = x/D
    req = (obstacle-frac*abs_b)/(1-frac)
    req[[0,-1]] = -np.inf
    idx = int(np.argmax(req))
    return max(0.0, float(req[idx]-ground_a)), idx

def solve_required_b(df, ground_b, abs_a, beam_d):
    D = float(df.distance_m.iloc[-1])
    x = df.distance_m.to_numpy()
    obstacle = df.terrain_elevation_m.to_numpy() + earth_bulge(x,D) + beam_d/2
    frac = x/D
    req = (obstacle-(1-frac)*abs_a)/frac
    req[[0,-1]] = -np.inf
    idx = int(np.argmax(req))
    return max(0.0, float(req[idx]-ground_b)), idx

def analyze(df, ground_a, ground_b, height_a, height_b, beam_d):
    D = float(df.distance_m.iloc[-1])
    x = df.distance_m.to_numpy()
    terrain = df.terrain_elevation_m.to_numpy()
    curvature = earth_bulge(x,D)
    abs_a, abs_b = ground_a+height_a, ground_b+height_b
    center = abs_a + (abs_b-abs_a)*(x/D)
    radius = beam_d/2
    lower, upper = center-radius, center+radius
    effective = terrain+curvature
    clearance = lower-effective
    test = clearance.copy()
    test[[0,-1]] = np.inf
    crit = int(np.argmin(test))
    req_a, idx_a = solve_required_a(df,ground_a,abs_b,beam_d)
    req_b, idx_b = solve_required_b(df,ground_b,abs_a,beam_d)
    base_line = center
    obstacle = effective+radius
    delta = obstacle-base_line
    delta[[0,-1]] = -np.inf
    equal_delta = max(0.0,float(np.max(delta)))
    out = df.copy()
    out["earth_curvature_m"] = curvature
    out["effective_terrain_m"] = effective
    out["centerline_los_m"] = center
    out["beam_lower_edge_m"] = lower
    out["beam_upper_edge_m"] = upper
    out["beam_clearance_m"] = clearance
    return dict(df=out, critical_idx=crit, critical_clearance=float(clearance[crit]),
                required_a=req_a, required_b=req_b,
                equal_a=height_a+equal_delta, equal_b=height_b+equal_delta,
                equal_delta=equal_delta, idx_a=idx_a, idx_b=idx_b)

st.title("📡 Velmenni LC LYNC™ LOS Feasibility V2")
st.caption("Distance • Azimuth • Terrain LOS • Required mounting height • Satellite visual inspection")

with st.sidebar:
    st.header("📍 Site A")
    lat_a = st.number_input("Latitude A", value=19.1533240, format="%.7f")
    lon_a = st.number_input("Longitude A", value=72.8509670, format="%.7f")
    height_a = st.number_input("Current height A (m AGL)", min_value=0.0, value=10.0, step=0.5)

    st.divider()
    st.header("📍 Site B")
    lat_b = st.number_input("Latitude B", value=19.1510150, format="%.7f")
    lon_b = st.number_input("Longitude B", value=72.8500800, format="%.7f")
    height_b = st.number_input("Current height B (m AGL)", min_value=0.0, value=10.0, step=0.5)

    st.divider()
    st.header("🔦 Optical planning")
    beam_d = st.select_slider("Maximum full beam diameter (m)", options=[2.0,2.5,3.0], value=3.0)
    st.caption("Focal length is not required. Set the maximum beam envelope your optical design targets.")
    samples = st.slider("Terrain samples", 50, 300, 120, 10)
    analyze_clicked = st.button("🔍 ANALYZE LINK", type="primary", use_container_width=True)

if analyze_clicked:
    st.session_state["los_analysis_requested"] = True

if st.session_state.get("los_analysis_requested", False):
    try:
        distance = haversine_distance(lat_a,lon_a,lat_b,lon_b)
        azimuth = initial_bearing(lat_a,lon_a,lat_b,lon_b)

        with st.spinner("Retrieving terrain and calculating LOS..."):
            df = terrain_profile(lat_a,lon_a,lat_b,lon_b,distance,samples)

        df["terrain_elevation_m"] = df["terrain_elevation_m"].interpolate().bfill().ffill()
        ground_a, ground_b = float(df.terrain_elevation_m.iloc[0]), float(df.terrain_elevation_m.iloc[-1])
        res = analyze(df,ground_a,ground_b,height_a,height_b,beam_d)
        df = res["df"]
        crit = df.iloc[res["critical_idx"]]

        st.subheader("📊 Link result")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Distance",f"{distance:.2f} m")
        c2.metric("Distance",f"{distance/1000:.3f} km")
        c3.metric("Azimuth A → B",f"{azimuth:.2f}°")
        c4.metric("Beam diameter",f"{beam_d:.1f} m")

        if res["critical_clearance"] >= 0:
            st.success(f"### 🟢 CURRENT BEAM PATH: CLEAR\nMinimum lower-edge clearance = {res['critical_clearance']:.2f} m")
        else:
            st.error(f"### 🔴 CURRENT BEAM PATH: BLOCKED\nMinimum lower-edge clearance = {res['critical_clearance']:.2f} m")

        st.subheader("🏗️ Required mounting height")
        h1,h2,h3 = st.columns(3)
        h1.metric("Raise Site A only",f"{res['required_a']:.1f} m AGL")
        h2.metric("Raise Site B only",f"{res['required_b']:.1f} m AGL")
        h3.metric("Raise both equally",f"{res['equal_a']:.1f} / {res['equal_b']:.1f} m AGL")

        st.caption("The height solver uses a conservative full-beam envelope: beam radius = selected maximum diameter / 2 along the path.")

        st.subheader("⚠️ Governing terrain point")
        g1,g2,g3,g4 = st.columns(4)
        g1.metric("Distance from A",f"{crit.distance_m:.1f} m")
        g2.metric("Terrain",f"{crit.terrain_elevation_m:.1f} m")
        g3.metric("Lower beam edge",f"{crit.beam_lower_edge_m:.1f} m")
        g4.metric("Clearance",f"{crit.beam_clearance_m:.2f} m")
        st.write(f"**Coordinates:** `{crit.latitude:.6f}, {crit.longitude:.6f}`")

        st.subheader("🛰️ Visual inspection map")
        fmap = folium.Map(location=[(lat_a+lat_b)/2,(lon_a+lon_b)/2],zoom_start=16,control_scale=True,tiles=None)
        folium.TileLayer("OpenStreetMap",name="Map").add_to(fmap)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",name="Satellite").add_to(fmap)
        folium.TileLayer(
            tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
            attr="OpenTopoMap",name="Topographic").add_to(fmap)
        folium.Marker([lat_a,lon_a],tooltip="SITE A",popup=f"Site A<br>Height {height_a:.1f} m AGL",
                     icon=folium.Icon(color="blue",icon="signal")).add_to(fmap)
        folium.Marker([lat_b,lon_b],tooltip="SITE B",popup=f"Site B<br>Height {height_b:.1f} m AGL",
                     icon=folium.Icon(color="red",icon="signal")).add_to(fmap)
        folium.PolyLine([[lat_a,lon_a],[lat_b,lon_b]],color="blue",weight=5,
                        tooltip=f"LC LYNC path: {distance:.1f} m / {azimuth:.1f}°").add_to(fmap)
        folium.CircleMarker([res["df"].iloc[res["critical_idx"]].latitude,res["df"].iloc[res["critical_idx"]].longitude],
                            radius=7,color="red",fill=True,fill_opacity=0.9,
                            tooltip="Governing terrain point").add_to(fmap)
        folium.LayerControl(collapsed=False).add_to(fmap)
        st_folium(fmap,use_container_width=True,height=560)

        gm = f"https://www.google.com/maps/dir/?api=1&origin={lat_a},{lon_a}&destination={lat_b},{lon_b}"
        st.markdown(f"[🌍 Open A → B path in Google Maps]({gm})")
        st.caption("In-app satellite imagery uses Esri World Imagery. Google Maps is provided for external visual cross-check. Satellite imagery alone does not reliably give object heights.")

        st.subheader("⛰️ Terrain / optical beam profile")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.distance_m,y=df.terrain_elevation_m,mode="lines",name="Terrain"))
        fig.add_trace(go.Scatter(x=df.distance_m,y=df.centerline_los_m,mode="lines",name="Optical centerline"))
        fig.add_trace(go.Scatter(x=df.distance_m,y=df.beam_lower_edge_m,mode="lines",name="Lower beam edge"))
        fig.add_trace(go.Scatter(x=df.distance_m,y=df.beam_upper_edge_m,mode="lines",name="Upper beam edge"))
        fig.add_trace(go.Scatter(x=[crit.distance_m],y=[crit.terrain_elevation_m],mode="markers",
                                 marker=dict(size=12),name="Governing point"))
        fig.update_layout(height=520,xaxis_title="Distance from Site A (m)",yaxis_title="Elevation (m)",
                          hovermode="x unified",legend=dict(orientation="h"))
        st.plotly_chart(fig,use_container_width=True)

        summary = pd.DataFrame([
            ["Distance (m)",distance],["Azimuth (deg)",azimuth],
            ["Ground elevation A (m)",ground_a],["Ground elevation B (m)",ground_b],
            ["Current A height AGL (m)",height_a],["Current B height AGL (m)",height_b],
            ["Maximum beam diameter (m)",beam_d],["Current minimum beam clearance (m)",res["critical_clearance"]],
            ["Required A height if B fixed (m AGL)",res["required_a"]],
            ["Required B height if A fixed (m AGL)",res["required_b"]],
            ["Equal-rise A height (m AGL)",res["equal_a"]],["Equal-rise B height (m AGL)",res["equal_b"]],
            ["Governing point distance from A (m)",crit.distance_m],
        ],columns=["Parameter","Value"])

        st.subheader("📌 Survey summary")
        st.dataframe(summary,use_container_width=True,hide_index=True)
        st.download_button("⬇️ Download LOS terrain CSV",df.to_csv(index=False).encode("utf-8"),
                           "LC_LYNC_LOS_V2_profile.csv","text/csv")

        st.warning("Preliminary planning only: visually/physically verify buildings, trees, poles, cranes and other structures. The 2–3 m beam is a planning envelope and should be checked against measured optical performance.")
    except requests.exceptions.RequestException as e:
        st.error("Elevation service could not be reached.")
        st.code(str(e))
    except Exception as e:
        st.error("Analysis failed.")
        st.exception(e)
else:
    st.info("Enter Site A/B coordinates and current heights, select the maximum 2–3 m beam diameter, then click ANALYZE LINK.")
