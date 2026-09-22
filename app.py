import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

try:
    #import srtm
except ImportError:
    srtm = None

st.set_page_config(page_title='LC LYNC LOS Feasibility', page_icon='📡', layout='wide')
st.title('📡 LC LYNC™ LiFi LOS Feasibility Calculator')
st.caption('Version 1.0 — preliminary terrain-based feasibility. Field verification is required.')

def haversine(lat1, lon1, lat2, lon2):
    R=6371000.0
    p1,p2=map(math.radians,[lat1,lat2]); dp=math.radians(lat2-lat1); dl=math.radians(lon2-lon1)
    a=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def bearing(lat1,lon1,lat2,lon2):
    p1,p2=map(math.radians,[lat1,lat2]); dl=math.radians(lon2-lon1)
    return (math.degrees(math.atan2(math.sin(dl)*math.cos(p2), math.cos(p1)*math.sin(p2)-math.sin(p1)*math.cos(p2)*math.cos(dl)))+360)%360

def destination(lat,lon,br_deg,d):
    R=6371000.; p1=math.radians(lat); l1=math.radians(lon); br=math.radians(br_deg); a=d/R
    p2=math.asin(math.sin(p1)*math.cos(a)+math.cos(p1)*math.sin(a)*math.cos(br))
    l2=l1+math.atan2(math.sin(br)*math.sin(a)*math.cos(p1),math.cos(a)-math.sin(p1)*math.sin(p2))
    return math.degrees(p2),(math.degrees(l2)+540)%360-180

def terrain_profile(lat1,lon1,lat2,lon2,D,n):
    if srtm is None: raise RuntimeError('srtm is not installed')
    svc=srtm.Srtm1HeightData() if hasattr(srtm,'Srtm1HeightData') else srtm.SrtmService()
    br=bearing(lat1,lon1,lat2,lon2); rows=[]
    for d in np.linspace(0,D,n):
        la,lo=destination(lat1,lon1,br,float(d))
        try:
            e=svc.get_elevation(float(la),float(lo))
        except Exception:
            e=None
        rows.append([d,la,lo,e])
    return pd.DataFrame(rows,columns=['distance_m','latitude','longitude','terrain_m'])

def analyze(df,h1,h2,wavelength_nm,fresnel_pct):
    df=df.dropna(subset=['terrain_m']).copy(); D=float(df.distance_m.iloc[-1]); x=df.distance_m.to_numpy()
    line=h1+(h2-h1)*(x/D)
    R=6371000.; earth_bulge=x*(D-x)/(2*R)
    terrain=df.terrain_m.to_numpy()-earth_bulge
    lam=wavelength_nm*1e-9; fz=np.sqrt(np.maximum(lam*x*(D-x)/D,0)); req=fz*fresnel_pct/100
    clear=line-terrain; clear_fz=clear-req
    df['los_centerline_m']=line; df['earth_bulge_m']=earth_bulge; df['first_fresnel_radius_m']=fz
    df['clearance_to_centerline_m']=clear; df['clearance_after_fresnel_m']=clear_fz
    mask=np.ones(len(df),dtype=bool); mask[[0,-1]]=False
    idx=int(np.argmin(np.where(mask,clear_fz,np.inf)))
    return df, float(clear[mask].min()), float(clear_fz[mask].min()), idx, float(fz.max())

with st.sidebar:
    st.header('Site A')
    lat1=st.number_input('Latitude A',value=6.9271,format='%.7f'); lon1=st.number_input('Longitude A',value=79.8612,format='%.7f')
    h1=st.number_input('Device height A above ground (m)',min_value=0.0,value=10.0,step=0.5)
    st.header('Site B')
    lat2=st.number_input('Latitude B',value=6.9350,format='%.7f'); lon2=st.number_input('Longitude B',value=79.8500,format='%.7f')
    h2=st.number_input('Device height B above ground (m)',min_value=0.0,value=10.0,step=0.5)
    st.header('Analysis')
    wavelength=st.number_input('Wavelength (nm)',min_value=100.0,max_value=2000.0,value=850.0,step=1.0)
    fresnel_pct=st.slider('Fresnel clearance criterion (%)',0,100,60)
    samples=st.slider('Terrain samples',50,1000,300,50)
    go_calc=st.button('🔍 Calculate Feasibility',type='primary',use_container_width=True)

if go_calc:
    if not (-90<=lat1<=90 and -180<=lon1<=180 and -90<=lat2<=90 and -180<=lon2<=180): st.error('Invalid coordinates.'); st.stop()
    D=haversine(lat1,lon1,lat2,lon2); br=bearing(lat1,lon1,lat2,lon2)
    try:
        with st.spinner('Loading terrain data and calculating LOS...'):
            prof=terrain_profile(lat1,lon1,lat2,lon2,D,samples)
            prof,los_min,fz_min,idx,max_fz=analyze(prof,h1,h2,wavelength,fresnel_pct)
    except Exception as e:
        st.error(f'Could not load terrain data: {e}')
        st.info('Install dependencies with: pip install -r requirements.txt. Internet access is needed on first terrain-data retrieval.')
        st.stop()
    c=st.columns(4); c[0].metric('Distance',f'{D/1000:.3f} km'); c[1].metric('Azimuth',f'{br:.1f}°'); c[2].metric('Min LOS clearance',f'{los_min:.2f} m'); c[3].metric('Max 1st Fresnel radius',f'{max_fz:.3f} m')
    if los_min>0 and fz_min>=0: st.success('PASS — terrain LOS and selected Fresnel criterion are clear.')
    elif los_min>0: st.warning('WARNING — geometric LOS is clear, but the selected Fresnel criterion is not met.')
    else: st.error('FAIL — terrain intersects the LOS centerline.')
    cp=prof.iloc[idx]
    st.write(f"**Critical terrain point:** {cp.distance_m:.1f} m from Site A — {cp.latitude:.6f}, {cp.longitude:.6f}; terrain ≈ {cp.terrain_m:.1f} m.")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=prof.distance_m,y=prof.terrain_m,mode='lines',name='Terrain'))
    fig.add_trace(go.Scatter(x=prof.distance_m,y=prof.los_centerline_m,mode='lines',name='LOS centerline'))
    fig.add_trace(go.Scatter(x=prof.distance_m,y=prof.los_centerline_m-prof.first_fresnel_radius_m*fresnel_pct/100,mode='lines',name=f'LOS - {fresnel_pct}% Fresnel'))
    fig.update_layout(title='Terrain / LOS / Fresnel Profile',xaxis_title='Distance from Site A (m)',yaxis_title='Elevation (m)',hovermode='x unified',height=500)
    st.plotly_chart(fig,use_container_width=True)
    st.subheader('Terrain / LOS Data')
    st.dataframe(prof.round(3),use_container_width=True)
    st.download_button('⬇️ Download CSV',prof.to_csv(index=False).encode(),file_name='lc_lync_los_profile.csv',mime='text/csv')
    st.info('This is a preliminary terrain-based calculator. Terrain DEM data may not detect trees, buildings, poles, cranes, wires, or temporary obstructions. For an 850 nm LiFi link, field verification and satellite/site imagery are required before installation.')
else:
    st.info('Enter Site A/B coordinates and device heights, then click Calculate Feasibility.')
    st.markdown('''### Version 1 features\n- GPS distance and azimuth\n- SRTM terrain profile\n- Straight optical LOS calculation\n- Earth-curvature correction\n- First Fresnel-zone calculation\n- Critical terrain point\n- PASS / WARNING / FAIL\n- CSV export\n\n### Planned Version 2\nMap/satellite layer, KML/KMZ export, manual building/tree obstruction input, automatic minimum mounting-height solver, and PDF feasibility report.''')
