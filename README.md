# LC LYNC™ LiFi LOS Feasibility Calculator — V1

## Run on Windows
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Run on Ubuntu / WSL
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

V1 calculates GPS distance, azimuth, terrain profile, terrain-based LOS, Earth-curvature correction, first Fresnel radius, critical terrain point, PASS/WARNING/FAIL and CSV export.

Important: this is preliminary terrain analysis. Trees, buildings, poles, cranes, wires and other optical obstructions may not appear in the DEM. Field verification is required for final LC LYNC installation approval.
