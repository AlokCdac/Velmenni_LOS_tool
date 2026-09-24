
import math, io, requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Velmenni LC LYNC LOS Feasibility V3", page_icon="📡", layout="wide")
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

def normalize_excel(df):
    # Accept common column naming variations from a manually prepared workbook.
    aliases = {
        "site_a_name":["site a name","site_a_name","site a","site_a"],
        "site_a_lat":["site a latitude","site a lat","lat a","latitude a","site_a_lat"],
        "site_a_lon":["site a longitude","site a long","lon a","longitude a","site_a_lon"],
        "site_b_name":["site b name","site_b_name","site b","site_b"],
        "site_b_lat":["site b latitude","site b lat","lat b","latitude b","site_b_lat"],
        "site_b_lon":["site b longitude","site b long","lon b","longitude b","site_b_lon"],
        "height_a":["height a","site a height","site_a_height"],
        "height_b":["height b","site b height","site_b_height"],
        "power_mode":["power mode","device power mode","power_mode"],
        "power_cable_length_m":["power cable length","power cable length m","power_cable_length_m"],
        "data_output":["data output","data output port","data_output"],
        "data_cable_length_m":["data cable length","data cable length m","data_cable_length_m"],
        "beam_diameter_m":["beam diameter","beam diameter m","beam_diameter_m"],
    }
    cols={str(c).strip().lower().replace("_"," "):c for c in df.columns}
    out=pd.DataFrame(index=df.index)
    for target, names in aliases.items():
        found=None
        for n in names:
            key=n.lower().replace("_"," ")
            if key in cols: found=cols[key]; break
        if found is not None: out[target]=df[found]
        else: out[target]=np.nan
    return out

def survey_template():
    return pd.DataFrame([{
        "Site A Name":"Site A-001","Site A Latitude":19.1533240,"Site A Longitude":72.8509670,
        "Site B Name":"Site B-001","Site B Latitude":19.1510150,"Site B Longitude":72.8500800,
        "Height A":10,"Height B":10,"Power Mode":"48V DC","Power Cable Length (m)":20,
        "Data Output":"ETH","Data Cable Length (m)":30,"Beam Diameter (m)":3.0
    }])

def make_report(rows):
    return pd.DataFrame(rows)

st.title("📡 Velmenni LC LYNC™ LOS Feasibility V3")
st.caption("Single-link feasibility + multi-survey Excel processing + visual survey selector")

with st.sidebar:
    st.header("📍 Site A")
    name_a=st.text_input("Site A name","Site A")
    lat_a=st.number_input("Latitude A",19.1533240,format="%.7f")
    lon_a=st.number_input("Longitude A",72.8509670,format="%.7f")
    ha=st.number_input("Current height A (m AGL)",0.0,value=10.0,step=0.5)
    st.divider()
    st.header("📍 Site B")
    name_b=st.text_input("Site B name","Site B")
    lat_b=st.number_input("Latitude B",19.1510150,format="%.7f")
    lon_b=st.number_input("Longitude B",72.8500800,format="%.7f")
    hb=st.number_input("Current height B (m AGL)",0.0,value=10.0,step=0.5)
    st.divider()
    st.header("⚡ Device / Cabling")
    power=st.selectbox("Device Power Mode",["48V DC","AC PoE","220V AC"])
    power_len=st.number_input("Power cable length (m)",min_value=0.0,value=20.0,step=1.0)
    data=st.selectbox("Data Output Port",["ETH","SMF (Single Mode Fibre)","MMF (Multimode Fibre)"])
    data_len=st.number_input("Data cable length (m)",min_value=0.0,value=30.0,step=1.0)
    st.caption("For AC PoE, the power cable field is treated as the PoE Ethernet cable length. For 48V DC / 220V AC it is the power cable length.")
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
        df,ci,reqa,reqb,eqa,eqb=calc(df,ha,hb,ga,gb,beam)
        crit=df.iloc[ci]

        c1,c2,c3,c4=st.columns(4)
        c1.metric("Distance",f"{D:.2f} m"); c2.metric("Azimuth A → B",f"{az:.2f}°")
        c3.metric("Beam diameter",f"{beam:.1f} m"); c4.metric("Minimum clearance",f"{crit.beam_clearance_m:.2f} m")
        st.success("### 🟢 CURRENT BEAM PATH: CLEAR") if crit.beam_clearance_m>=0 else st.error("### 🔴 CURRENT BEAM PATH: BLOCKED")

        st.subheader("🏗️ Required mounting height")
        q1,q2,q3=st.columns(3)
        q1.metric("Raise Site A only",f"{reqa:.1f} m AGL")
        q2.metric("Raise Site B only",f"{reqb:.1f} m AGL")
        q3.metric("Raise both equally",f"{eqa:.1f} / {eqb:.1f} m AGL")

        st.subheader("🛰️ Visual inspection map")
        m=folium.Map(location=[(lat_a+lat_b)/2,(lon_a+lon_b)/2],zoom_start=16,tiles=None)
        folium.TileLayer("OpenStreetMap",name="Map").add_to(m)
        folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                         attr="Esri World Imagery",name="Satellite").add_to(m)
        folium.TileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                         attr="OpenTopoMap",name="Topographic").add_to(m)
        folium.Marker([lat_a,lon_a],tooltip=name_a,popup=f"<b>{name_a}</b><br>Lat: {lat_a:.7f}<br>Long: {lon_a:.7f}<br>Height: {ha:.1f} m AGL",
                      icon=folium.Icon(color="blue",icon="signal")).add_to(m)
        folium.Marker([lat_b,lon_b],tooltip=name_b,popup=f"<b>{name_b}</b><br>Lat: {lat_b:.7f}<br>Long: {lon_b:.7f}<br>Height: {hb:.1f} m AGL",
                      icon=folium.Icon(color="red",icon="signal")).add_to(m)
        folium.PolyLine([[lat_a,lon_a],[lat_b,lon_b]],color="blue",weight=5,
                        tooltip=f"{D:.1f} m | Azimuth {az:.1f}°").add_to(m)
        for lat,lon,text,color in [(lat_a,lon_a,name_a,"#0b3d91"),(lat_b,lon_b,name_b,"#a40000")]:
            folium.Marker([lat,lon],icon=folium.DivIcon(html=f"""<div style="font-size:13px;font-weight:700;color:{color};white-space:nowrap;background:rgba(255,255,255,.9);border:1px solid {color};border-radius:4px;padding:3px 6px;transform:translate(10px,-34px)">{text}</div>""")).add_to(m)
        folium.Marker([(lat_a+lat_b)/2,(lon_a+lon_b)/2],icon=folium.DivIcon(html=f"""<div style="font-size:13px;font-weight:700;color:#111;white-space:nowrap;background:rgba(255,255,255,.94);border:1px solid #333;border-radius:5px;padding:4px 8px;transform:translate(-50%,-50%);text-align:center">{D:.1f} m<br><span style="font-size:11px">Azimuth {az:.1f}°</span></div>""")).add_to(m)
        folium.CircleMarker([crit.latitude,crit.longitude],radius=7,color="red",fill=True,tooltip="Governing terrain point").add_to(m)
        folium.LayerControl().add_to(m)
        st_folium(m,use_container_width=True,height=560)
        st.markdown(f"[🌍 Open A → B path in Google Maps](https://www.google.com/maps/dir/?api=1&origin={lat_a},{lon_a}&destination={lat_b},{lon_b})")

        st.subheader("⛰️ Terrain / optical beam profile")
        fig=go.Figure()
        for col,label in [("terrain_elevation_m","Terrain"),("centerline_los_m","Optical centerline"),("beam_lower_edge_m","Lower beam edge"),("beam_upper_edge_m","Upper beam edge")]:
            fig.add_trace(go.Scatter(x=df.distance_m,y=df[col],mode="lines",name=label))
        fig.add_trace(go.Scatter(x=[crit.distance_m],y=[crit.terrain_elevation_m],mode="markers",marker=dict(size=12),name="Governing point"))
        fig.update_layout(height=500,xaxis_title="Distance from Site A (m)",yaxis_title="Elevation (m)",hovermode="x unified",legend=dict(orientation="h"))
        st.plotly_chart(fig,use_container_width=True)

        st.subheader("📋 Survey report")
        report=pd.DataFrame([
            ["Site A Name",name_a],["Site A Latitude",lat_a],["Site A Longitude",lon_a],
            ["Site B Name",name_b],["Site B Latitude",lat_b],["Site B Longitude",lon_b],
            ["Device Power Mode",power],
            ["Power / PoE Cable Length (m)",power_len],
            ["Data Output Port",data],
            ["Data Cable Length (m)",data_len],
            ["Distance (m)",D],["Azimuth A → B (deg)",az],
            ["Ground Elevation A (m)",ga],["Ground Elevation B (m)",gb],
            ["Current A Height AGL (m)",ha],["Current B Height AGL (m)",hb],
            ["Maximum Beam Diameter (m)",beam],["Minimum Beam Clearance (m)",crit.beam_clearance_m],
            ["Required A Height if B Fixed (m AGL)",reqa],["Required B Height if A Fixed (m AGL)",reqb],
            ["Equal-rise A Height (m AGL)",eqa],["Equal-rise B Height (m AGL)",eqb],
            ["Governing Point Distance from A (m)",crit.distance_m],
            ["Governing Point Latitude",crit.latitude],["Governing Point Longitude",crit.longitude],
        ],columns=["Parameter","Value"])
        st.dataframe(report,use_container_width=True,hide_index=True)
        st.download_button("⬇️ Download Single Survey CSV",report.to_csv(index=False).encode(),"LC_LYNC_Single_Survey.csv","text/csv")

        st.warning("Preliminary planning only: visually/physically verify buildings, trees, poles and other structures. The selected 2–3 m beam is a planning envelope and should be checked against measured optical performance.")
    except Exception as e:
        st.error("Analysis failed"); st.exception(e)
else:
    st.info("Enter the two sites, cabling and optical parameters, then click ANALYZE LINK.")

st.divider()
st.header("📚 Multi-Survey Excel Processor")
st.write("Upload an Excel file containing multiple survey rows. The tool calculates distance, azimuth, terrain/LOS and required heights for every row, then produces an Excel survey report.")

template = survey_template()
st.download_button("⬇️ Download Excel input template",template.to_excel(index=False,engine="openpyxl"),"LC_LYNC_Multi_Survey_Template.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

up=st.file_uploader("Upload multi-survey Excel (.xlsx)",type=["xlsx"])
if up:
    try:
        raw=pd.read_excel(up)
        inp=normalize_excel(raw)
        required=["site_a_name","site_a_lat","site_a_lon","site_b_name","site_b_lat","site_b_lon"]
        missing=[x for x in required if inp[x].isna().all()]
        if missing:
            st.error("Missing required fields: "+", ".join(missing))
        else:
            st.success(f"Loaded {len(inp)} survey row(s).")
            labels=[]
            results=[]
            profiles={}
            for i,row in inp.iterrows():
                try:
                    A=str(row.site_a_name) if pd.notna(row.site_a_name) else f"Survey {i+1} A"
                    B=str(row.site_b_name) if pd.notna(row.site_b_name) else f"Survey {i+1} B"
                    la=float(row.site_a_lat); loa=float(row.site_a_lon); lb=float(row.site_b_lat); lob=float(row.site_b_lon)
                    haa=float(row.height_a) if pd.notna(row.height_a) else 10.0
                    hbb=float(row.height_b) if pd.notna(row.height_b) else 10.0
                    bm=float(row.beam_diameter_m) if pd.notna(row.beam_diameter_m) else 3.0
                    pm=str(row.power_mode) if pd.notna(row.power_mode) else "48V DC"
                    pc=float(row.power_cable_length_m) if pd.notna(row.power_cable_length_m) else 0.0
                    do=str(row.data_output) if pd.notna(row.data_output) else "ETH"
                    dc=float(row.data_cable_length_m) if pd.notna(row.data_cable_length_m) else 0.0
                    D,az,p=profile(la,loa,lb,lob,100)
                    p["terrain_elevation_m"]=p.terrain_elevation_m.interpolate().bfill().ffill()
                    ga2,gb2=float(p.terrain_elevation_m.iloc[0]),float(p.terrain_elevation_m.iloc[-1])
                    p,ci2,ra2,rb2,eqa2,eqb2=calc(p,haa,hbb,ga2,gb2,bm); cr=p.iloc[ci2]
                    status="CLEAR" if cr.beam_clearance_m>=0 else "BLOCKED"
                    results.append({"Survey ID":i+1,"Site A Name":A,"Site A Latitude":la,"Site A Longitude":loa,
                        "Site B Name":B,"Site B Latitude":lb,"Site B Longitude":lob,"Device Power Mode":pm,
                        "Power/PoE Cable Length (m)":pc,"Data Output Port":do,"Data Cable Length (m)":dc,
                        "Distance (m)":D,"Azimuth (deg)":az,"Height A (m AGL)":haa,"Height B (m AGL)":hbb,
                        "Beam Diameter (m)":bm,"LOS Status":status,"Minimum Beam Clearance (m)":float(cr.beam_clearance_m),
                        "Required A Height (m AGL)":ra2,"Required B Height (m AGL)":rb2,
                        "Equal-rise A Height (m AGL)":eqa2,"Equal-rise B Height (m AGL)":eqb2,
                        "Governing Point Distance (m)":float(cr.distance_m),"Governing Point Latitude":float(cr.latitude),
                        "Governing Point Longitude":float(cr.longitude)})
                    labels.append(f"{i+1}: {A} → {B} ({D:.1f} m)")
                    profiles[i+1]=(p,A,B,la,loa,lb,lob,D,az,haa,hbb,pm,pc,do,dc)
                except Exception as ex:
                    results.append({"Survey ID":i+1,"Site A Name":row.site_a_name,"Site B Name":row.site_b_name,"LOS Status":"ERROR","Error":str(ex)})
                    labels.append(f"{i+1}: ERROR")
            results_df=pd.DataFrame(results)
            st.subheader("📊 Multi-survey report")
            st.dataframe(results_df,use_container_width=True,hide_index=True)
            buf=io.BytesIO()
            with pd.ExcelWriter(buf,engine="openpyxl") as writer:
                results_df.to_excel(writer,index=False,sheet_name="Survey Report")
                inp.to_excel(writer,index=False,sheet_name="Input Data")
            st.download_button("⬇️ Download Multi-Survey Excel Report",buf.getvalue(),
                               "LC_LYNC_Multi_Survey_Report.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            if profiles:
                st.subheader("👁️ Visual check one survey at a time")
                choice=st.selectbox("Select survey to view",labels)
                sid=int(choice.split(":")[0])
                p,A,B,la,loa,lb,lob,D,az,haa,hbb,pm,pc,do,dc=profiles[sid]
                vm=folium.Map(location=[(la+lb)/2,(loa+lob)/2],zoom_start=16,tiles=None)
                folium.TileLayer("OpenStreetMap",name="Map").add_to(vm)
                folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                                 attr="Esri World Imagery",name="Satellite").add_to(vm)
                folium.Marker([la,loa],tooltip=A,icon=folium.Icon(color="blue",icon="signal")).add_to(vm)
                folium.Marker([lb,lob],tooltip=B,icon=folium.Icon(color="red",icon="signal")).add_to(vm)
                folium.PolyLine([[la,loa],[lb,lob]],color="blue",weight=5,tooltip=f"{D:.1f} m | {az:.1f}°").add_to(vm)
                folium.Marker([(la+lb)/2,(loa+lob)/2],icon=folium.DivIcon(html=f"""<div style="font-weight:700;background:white;border:1px solid #333;padding:4px;transform:translate(-50%,-50%)">{D:.1f} m<br><small>Azimuth {az:.1f}°</small></div>""")).add_to(vm)
                folium.LayerControl().add_to(vm)
                st_folium(vm,use_container_width=True,height=520,key=f"multi_map_{sid}")
                selected=results_df[results_df["Survey ID"]==sid]
                st.dataframe(selected.T.rename(columns={selected.index[0]:"Value"}) if len(selected) else pd.DataFrame(),use_container_width=True)
    except Exception as ex:
        st.error(f"Could not process Excel: {ex}")
