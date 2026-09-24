
import math, requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Velmenni LC LYNC LOS V2", page_icon="📡", layout="wide")
R = 6371000.0

def hav(a,b,c,d):
    p1,p2=map(math.radians,[a,c]); dp=math.radians(c-a); dl=math.radians(d-b)
    x=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.atan2(math.sqrt(x),math.sqrt(1-x))

def bearing(a,b,c,d):
    p1,p2=map(math.radians,[a,c]); dl=math.radians(d-b)
    x=math.sin(dl)*math.cos(p2)
    y=math.cos(p1)*math.sin(p2)-math.sin(p1)*math.cos(p2)*math.cos(dl)
    return (math.degrees(math.atan2(x,y))+360)%360

def point(lat,lon,b,dist):
    p,l,br=map(math.radians,[lat,lon,b]); q=dist/R
    la=math.asin(math.sin(p)*math.cos(q)+math.cos(p)*math.sin(q)*math.cos(br))
    lo=l+math.atan2(math.sin(br)*math.sin(q)*math.cos(p),math.cos(q)-math.sin(p)*math.sin(la))
    return math.degrees(la),(math.degrees(lo)+540)%360-180

@st.cache_data(ttl=3600)
def elev(coords):
    r=requests.get("https://api.open-meteo.com/v1/elevation",
                   params={"latitude":",".join(f"{x[0]:.7f}" for x in coords),
                           "longitude":",".join(f"{x[1]:.7f}" for x in coords)},timeout=60)
    r.raise_for_status(); j=r.json()
    if isinstance(j,list): return [float(x["elevation"]) for x in j]
    return [float(x) for x in j["elevation"]]

def profile(a,b,c,d,n):
    D=hav(a,b,c,d); br=bearing(a,b,c,d); ds=np.linspace(0,D,n)
    pts=[point(a,b,br,float(x)) for x in ds]
    es=[]
    for i in range(0,n,100): es += elev(pts[i:i+100])
    return D,br,pd.DataFrame({"distance_m":ds,"latitude":[x[0] for x in pts],
        "longitude":[x[1] for x in pts],"terrain_elevation_m":es})

def calc(df,ha,hb,ga,gb,beam):
    D=float(df.distance_m.iloc[-1]); x=df.distance_m.to_numpy(); t=df.terrain_elevation_m.to_numpy()
    bulge=x*(D-x)/(2*R); A=ga+ha; B=gb+hb
    center=A+(B-A)*x/D; lower=center-beam/2; upper=center+beam/2
    clear=lower-(t+bulge); test=clear.copy(); test[[0,-1]]=np.inf; ci=int(np.argmin(test))
    obs=t+bulge+beam/2; f=x/D
    ra=(obs-f*B)/(1-f); ra[[0,-1]]=-np.inf; ia=int(np.argmax(ra))
    rb=(obs-(1-f)*A)/f; rb[[0,-1]]=-np.inf; ib=int(np.argmax(rb))
    delta=obs-center; delta[[0,-1]]=-np.inf; de=max(0,float(np.max(delta)))
    out=df.copy(); out["earth_curvature_m"]=bulge; out["centerline_los_m"]=center
    out["beam_lower_edge_m"]=lower; out["beam_upper_edge_m"]=upper; out["beam_clearance_m"]=clear
    return out,ci,max(0,float(ra[ia]-ga)),max(0,float(rb[ib]-gb)),ha+de,hb+de

st.title("📡 Velmenni LC LYNC™ LOS Feasibility V2")
st.caption("Distance • Azimuth • Required mounting height • Terrain LOS • Satellite visual inspection")

with st.sidebar:
    st.header("📍 Site A")
    name_a=st.text_input("Site A name","Site A")
    lat_a=st.number_input("Latitude A",19.1533240,format="%.7f")
    lon_a=st.number_input("Longitude A",72.8509670,format="%.7f")
    ha=st.number_input("Current height A (m AGL)",0.0, value=10.0, step=0.5)
    st.divider()
    st.header("📍 Site B")
    name_b=st.text_input("Site B name","Site B")
    lat_b=st.number_input("Latitude B",19.1510150,format="%.7f")
    lon_b=st.number_input("Longitude B",72.8500800,format="%.7f")
    hb=st.number_input("Current height B (m AGL)",0.0, value=10.0, step=0.5)
    st.divider()
    st.header("⚡ Device / Survey")
    power=st.selectbox("Device Power Mode",["48V DC","AC PoE","220V AC"])
    data=st.selectbox("Data Output Port",["ETH","SMF (Single Mode Fibre)","MMF (Multimode Fibre)"])
    st.divider()
    beam=st.select_slider("Maximum full beam diameter (m)",options=[2.0,2.5,3.0],value=3.0)
    n=st.slider("Terrain samples",50,300,120,10)
    click=st.button("🔍 ANALYZE LINK",type="primary",use_container_width=True)

if click: st.session_state["run"]=True
if st.session_state.get("run",False):
    try:
        D,az,df=profile(lat_a,lon_a,lat_b,lon_b,n)
        df["terrain_elevation_m"]=df.terrain_elevation_m.interpolate().bfill().ffill()
        ga,gb=float(df.terrain_elevation_m.iloc[0]),float(df.terrain_elevation_m.iloc[-1])
        df,ci,reqa,reqb,eqa,eqb=calc(df,ha,hb,ga,gb,beam); crit=df.iloc[ci]

        c1,c2,c3,c4=st.columns(4)
        c1.metric("Distance",f"{D:.2f} m"); c2.metric("Azimuth A → B",f"{az:.2f}°")
        c3.metric("Beam diameter",f"{beam:.1f} m"); c4.metric("Minimum clearance",f"{crit.beam_clearance_m:.2f} m")
        if crit.beam_clearance_m>=0: st.success("### 🟢 CURRENT BEAM PATH: CLEAR")
        else: st.error("### 🔴 CURRENT BEAM PATH: BLOCKED")

        st.subheader("🏗️ Required mounting height")
        x1,x2,x3=st.columns(3)
        x1.metric("Raise Site A only",f"{reqa:.1f} m AGL")
        x2.metric("Raise Site B only",f"{reqb:.1f} m AGL")
        x3.metric("Raise both equally",f"{eqa:.1f} / {eqb:.1f} m AGL")

        st.subheader("🛰️ Visual inspection map")
        m=folium.Map(location=[(lat_a+lat_b)/2,(lon_a+lon_b)/2],zoom_start=16,tiles=None)
        folium.TileLayer("OpenStreetMap",name="Map").add_to(m)
        folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                         attr="Esri World Imagery",name="Satellite").add_to(m)
        folium.TileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                         attr="OpenTopoMap",name="Topographic").add_to(m)
        folium.Marker([lat_a,lon_a],tooltip=name_a,
                      popup=f"<b>{name_a}</b><br>Height: {ha:.1f} m AGL",
                      icon=folium.Icon(color="blue",icon="signal")).add_to(m)
        folium.Marker([lat_b,lon_b],tooltip=name_b,
                      popup=f"<b>{name_b}</b><br>Height: {hb:.1f} m AGL",
                      icon=folium.Icon(color="red",icon="signal")).add_to(m)
        folium.PolyLine([[lat_a,lon_a],[lat_b,lon_b]],color="blue",weight=5,
                        tooltip=f"{D:.1f} m | Azimuth {az:.1f}°").add_to(m)

        # Persistent endpoint labels
        for lat,lon,text,color,dx in [(lat_a,lon_a,name_a,"#0b3d91","10px"),
                                       (lat_b,lon_b,name_b,"#a40000","10px")]:
            folium.Marker([lat,lon],icon=folium.DivIcon(html=f"""
            <div style="font-size:13px;font-weight:700;color:{color};white-space:nowrap;
            background:rgba(255,255,255,.90);border:1px solid {color};border-radius:4px;
            padding:3px 6px;transform:translate({dx},-34px)">{text}</div>""")).add_to(m)

        # Persistent link-distance/azimuth label at midpoint
        mid=((lat_a+lat_b)/2,(lon_a+lon_b)/2)
        folium.Marker(mid,icon=folium.DivIcon(html=f"""
        <div style="font-size:13px;font-weight:700;color:#111;white-space:nowrap;
        background:rgba(255,255,255,.94);border:1px solid #333;border-radius:5px;
        padding:4px 8px;transform:translate(-50%,-50%);text-align:center">
        {D:.1f} m<br><span style="font-size:11px">Azimuth {az:.1f}°</span></div>""")).add_to(m)
        folium.CircleMarker([crit.latitude,crit.longitude],radius=7,color="red",fill=True,
                            tooltip="Governing terrain point").add_to(m)
        folium.LayerControl().add_to(m)
        st_folium(m,use_container_width=True,height=560)
        st.markdown(f"[🌍 Open A → B path in Google Maps](https://www.google.com/maps/dir/?api=1&origin={lat_a},{lon_a}&destination={lat_b},{lon_b})")

        st.subheader("⛰️ Terrain / optical beam profile")
        fig=go.Figure()
        for col,label in [("terrain_elevation_m","Terrain"),("centerline_los_m","Optical centerline"),
                          ("beam_lower_edge_m","Lower beam edge"),("beam_upper_edge_m","Upper beam edge")]:
            fig.add_trace(go.Scatter(x=df.distance_m,y=df[col],mode="lines",name=label))
        fig.add_trace(go.Scatter(x=[crit.distance_m],y=[crit.terrain_elevation_m],mode="markers",
                                 marker=dict(size=12),name="Governing point"))
        fig.update_layout(height=500,xaxis_title="Distance from Site A (m)",yaxis_title="Elevation (m)",
                          hovermode="x unified",legend=dict(orientation="h"))
        st.plotly_chart(fig,use_container_width=True)

        st.subheader("📋 Survey report")
        s1,s2,s3,s4=st.columns(4)
        s1.metric("Site A",name_a); s2.metric("Site B",name_b)
        s3.metric("Power",power); s4.metric("Data",data)
        report=pd.DataFrame([
            ["Site A name",name_a],["Site B name",name_b],
            ["Device Power Mode",power],["Data Output Port",data],
            ["Distance (m)",D],["Azimuth A → B (deg)",az],
            ["Ground elevation A (m)",ga],["Ground elevation B (m)",gb],
            ["Current A height AGL (m)",ha],["Current B height AGL (m)",hb],
            ["Maximum beam diameter (m)",beam],["Minimum beam clearance (m)",crit.beam_clearance_m],
            ["Required A height if B fixed (m AGL)",reqa],
            ["Required B height if A fixed (m AGL)",reqb],
            ["Equal-rise A height (m AGL)",eqa],["Equal-rise B height (m AGL)",eqb],
            ["Governing point distance from A (m)",crit.distance_m],
        ],columns=["Parameter","Value"])
        st.dataframe(report,use_container_width=True,hide_index=True)
        st.download_button("⬇️ Download LOS terrain CSV",df.to_csv(index=False).encode(),"LC_LYNC_LOS_V2_profile.csv","text/csv")

        st.warning("Preliminary planning only: visually/physically verify buildings, trees, poles and other structures. The selected 2–3 m beam is a planning envelope and should be checked against measured optical performance.")
        if st.button("🧹 Clear analysis and start a new link",use_container_width=True):
            st.session_state["run"]=False; st.rerun()
    except Exception as e:
        st.error("Analysis failed"); st.exception(e)
else:
    st.info("Enter the two site names/coordinates, current heights, power mode and data output, then click ANALYZE LINK.")
