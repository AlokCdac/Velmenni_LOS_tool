import math
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Velmenni LC LYNC LOS Feasibility",
    page_icon="📡",
    layout="wide",
)

st.title("📡 Velmenni LC LYNC™ LiFi LOS Feasibility Tool")
st.caption(
    "Preliminary terrain-based feasibility analysis for optical wireless / LiFi links"
)


# ============================================================
# CONSTANTS
# ============================================================

EARTH_RADIUS_M = 6_371_000.0


# ============================================================
# GEOGRAPHICAL FUNCTIONS
# ============================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two GPS coordinates.
    Returns distance in metres.
    """

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return EARTH_RADIUS_M * c


def initial_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate initial bearing from Site A to Site B.
    """

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    dlon = math.radians(lon2 - lon1)

    x = math.sin(dlon) * math.cos(lat2_rad)

    y = (
        math.cos(lat1_rad) * math.sin(lat2_rad)
        - math.sin(lat1_rad)
        * math.cos(lat2_rad)
        * math.cos(dlon)
    )

    bearing = math.degrees(math.atan2(x, y))

    return (bearing + 360) % 360


def destination_point(lat, lon, bearing_deg, distance_m):
    """
    Calculate a GPS point at a given distance and bearing.
    """

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing_deg)

    angular_distance = distance_m / EARTH_RADIUS_M

    new_lat = math.asin(
        math.sin(lat_rad) * math.cos(angular_distance)
        + math.cos(lat_rad)
        * math.sin(angular_distance)
        * math.cos(bearing_rad)
    )

    new_lon = lon_rad + math.atan2(
        math.sin(bearing_rad)
        * math.sin(angular_distance)
        * math.cos(lat_rad),
        math.cos(angular_distance)
        - math.sin(lat_rad) * math.sin(new_lat),
    )

    new_lon = (math.degrees(new_lon) + 540) % 360 - 180

    return math.degrees(new_lat), new_lon


# ============================================================
# ELEVATION API
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_elevations(points):
    """
    Retrieve elevation values using the Open-Meteo Elevation API.
    points: list of (latitude, longitude)
    """
    lats = [float(lat) for lat, _ in points]
    lons = [float(lon) for _, lon in points]

    url = "https://api.open-meteo.com/v1/elevation"
    params = {
        "latitude": ",".join(map(str, lats)),
        "longitude": ",".join(map(str, lons)),
    }

    response = requests.get(
        url,
        params=params,
        timeout=60,
    )

    response.raise_for_status()
    data = response.json()

    if "elevation" not in data:
        raise RuntimeError("Elevation API returned an unexpected response.")

    return [float(e) if e is not None else np.nan for e in data["elevation"]]


def get_terrain_profile(
    lat1,
    lon1,
    lat2,
    lon2,
    distance_m,
    sample_points,
):
    """
    Generate intermediate GPS points and obtain terrain elevation.
    """

    bearing = initial_bearing(
        lat1,
        lon1,
        lat2,
        lon2,
    )

    distances = np.linspace(
        0,
        distance_m,
        sample_points,
    )

    coordinates = []

    for distance in distances:
        lat, lon = destination_point(
            lat1,
            lon1,
            bearing,
            float(distance),
        )

        coordinates.append(
            (lat, lon)
        )

    # Keep requests reasonably sized.
    batch_size = 100

    elevations = []

    for start in range(0, len(coordinates), batch_size):

        batch = coordinates[
            start:start + batch_size
        ]

        batch_elevations = get_elevations(batch)

        elevations.extend(batch_elevations)

        # Small delay between API batches.
        if start + batch_size < len(coordinates):
            time.sleep(0.2)

    df = pd.DataFrame(
        {
            "distance_m": distances,
            "latitude": [x[0] for x in coordinates],
            "longitude": [x[1] for x in coordinates],
            "terrain_elevation_m": elevations,
        }
    )

    return df


# ============================================================
# FRESNEL CALCULATION
# ============================================================

def calculate_fresnel_radius(
    distance_m,
    wavelength_m,
    distance_from_a,
):
    """
    First Fresnel-zone radius.

    r = sqrt(lambda * d1 * d2 / D)
    """

    d1 = distance_from_a
    d2 = distance_m - distance_from_a

    if d1 <= 0 or d2 <= 0:
        return 0.0

    return math.sqrt(
        wavelength_m
        * d1
        * d2
        / distance_m
    )


# ============================================================
# LOS ANALYSIS
# ============================================================

def analyze_los(
    terrain_df,
    device_height_a,
    device_height_b,
    wavelength_nm,
    fresnel_percentage,
):
    """
    Analyze terrain versus optical LOS.
    """

    df = terrain_df.copy()

    distance_m = float(
        df["distance_m"].iloc[-1]
    )

    terrain = df[
        "terrain_elevation_m"
    ].to_numpy()

    x = df[
        "distance_m"
    ].to_numpy()

    # Earth curvature
    earth_bulge = (
        x
        * (distance_m - x)
        / (2 * EARTH_RADIUS_M)
    )

    # Endpoint ground elevations
    ground_a = float(terrain[0])
    ground_b = float(terrain[-1])

    # Actual optical aperture elevations
    aperture_a = ground_a + device_height_a
    aperture_b = ground_b + device_height_b

    # Straight optical LOS
    los_absolute = (
        aperture_a
        + (aperture_b - aperture_a)
        * (x / distance_m)
    )

    # Correct terrain for earth curvature.
    effective_terrain = terrain + earth_bulge

    # Geometrical LOS clearance
    geometric_clearance = los_absolute - effective_terrain

    # Fresnel zone
    wavelength_m = wavelength_nm * 1e-9
    fresnel_radius = np.zeros_like(x)

    for i, distance in enumerate(x):
        fresnel_radius[i] = calculate_fresnel_radius(
            distance_m,
            wavelength_m,
            float(distance),
        )

    required_fresnel_clearance = (
        fresnel_radius
        * fresnel_percentage
        / 100.0
    )

    clearance_after_fresnel = (
        geometric_clearance
        - required_fresnel_clearance
    )

    # Ignore endpoints when looking for obstruction.
    interior = np.ones(len(df), dtype=bool)
    interior[0] = False
    interior[-1] = False

    interior_clearance = np.where(
        interior,
        geometric_clearance,
        np.inf,
    )

    interior_fresnel_clearance = np.where(
        interior,
        clearance_after_fresnel,
        np.inf,
    )
# ============================================================
# RECOMMENDED MOUNTING HEIGHT CALCULATION
# ============================================================

def calculate_min_mounting_heights(terrain_df, total_distance_m, clearance_margin_m=1.0):
    """
    Calculates the minimum equal mounting height required at both sites
    to clear all terrain obstructions with a safety margin.
    """
    df = terrain_df.copy()
    x = df["distance_m"].to_numpy()
    terrain = df["terrain_elevation_m"].to_numpy()
    
    # Calculate Earth curvature
    earth_bulge = x * (total_distance_m - x) / (2 * EARTH_RADIUS_M)
    effective_terrain = terrain + earth_bulge
    
    ground_a = terrain[0]
    ground_b = terrain[-1]
    
    # Exclude immediate endpoints (the sites themselves)
    x_mid = x[1:-1]
    eff_mid = effective_terrain[1:-1] + clearance_margin_m
    
    # Linear interpolation factor across path
    fraction = x_mid / total_distance_m
    
    # Elevation of imaginary ground-to-ground line
    ground_path = ground_a + fraction * (ground_b - ground_a)
    
    # Calculate height deficit along intermediate terrain
    height_deficit = eff_mid - ground_path
    
    # Get maximum deficit needed
    min_mount_height = max(0.0, float(np.max(height_deficit)))
    
    return min_mount_height

    critical_los_index = int(np.argmin(interior_clearance))
    critical_fresnel_index = int(np.argmin(interior_fresnel_clearance))

    minimum_los_clearance = float(geometric_clearance[critical_los_index])
    minimum_fresnel_clearance = float(clearance_after_fresnel[critical_fresnel_index])

    # Add calculated values to dataframe
    df["earth_curvature_m"] = earth_bulge
    df["effective_terrain_m"] = effective_terrain
    df["los_elevation_m"] = los_absolute
    df["geometric_clearance_m"] = geometric_clearance
    df["fresnel_radius_m"] = fresnel_radius
    df["required_fresnel_clearance_m"] = required_fresnel_clearance
    df["clearance_after_fresnel_m"] = clearance_after_fresnel

    return {
        "data": df,
        "ground_a": ground_a,
        "ground_b": ground_b,
        "aperture_a": aperture_a,
        "aperture_b": aperture_b,
        "minimum_los_clearance": minimum_los_clearance,
        "minimum_fresnel_clearance": minimum_fresnel_clearance,
        "critical_los_index": critical_los_index,
        "critical_fresnel_index": critical_fresnel_index,
        "critical_los_distance": float(x[critical_los_index]),
        "critical_fresnel_distance": float(x[critical_fresnel_index]),
        "critical_los_lat": float(df.iloc[critical_los_index]["latitude"]),
        "critical_los_lon": float(df.iloc[critical_los_index]["longitude"]),
        "max_fresnel_radius": float(np.max(fresnel_radius)),
    }


# ============================================================
# SIDEBAR INPUTS
# ============================================================

with st.sidebar:

    st.header("📍 Site A")

    latitude_a = st.number_input(
        "Latitude A",
        value=6.922819,
        format="%.7f",
    )

    longitude_a = st.number_input(
        "Longitude A",
        value=79.853370,
        format="%.7f",
    )

    height_a = st.number_input(
        "LC LYNC device height A above ground (m)",
        min_value=0.0,
        value=10.0,
        step=0.5,
    )

    st.divider()

    st.header("📍 Site B")

    latitude_b = st.number_input(
        "Latitude B",
        value=6.935177,
        format="%.7f",
    )

    longitude_b = st.number_input(
        "Longitude B",
        value=79.853672,
        format="%.7f",
    )

    height_b = st.number_input(
        "LC LYNC device height B above ground (m)",
        min_value=0.0,
        value=10.0,
        step=0.5,
    )

    st.divider()

    st.header("📡 Optical Parameters")

    wavelength_nm = st.number_input(
        "Optical wavelength (nm)",
        min_value=100.0,
        max_value=2000.0,
        value=850.0,
        step=1.0,
    )

    beam_divergence = st.number_input(
        "Beam divergence (°)",
        min_value=0.01,
        max_value=30.0,
        value=1.0,
        step=0.1,
    )

    optical_aperture = st.number_input(
        "Optical aperture diameter (mm)",
        min_value=1.0,
        max_value=1000.0,
        value=100.0,
        step=1.0,
    )

    st.divider()

    st.header("📐 Analysis")

    fresnel_percentage = st.slider(
        "Required first Fresnel clearance (%)",
        min_value=0,
        max_value=100,
        value=60,
        step=5,
    )

    sample_points = st.slider(
        "Terrain sample points",
        min_value=30,
        max_value=300,
        value=100,
        step=10,
    )

    calculate = st.button(
        "🔍 ANALYZE LINK",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

if calculate:

    try:

        # ----------------------------------------------------
        # Distance & Bearing
        # ----------------------------------------------------

        distance_m = haversine_distance(
            latitude_a,
            longitude_a,
            latitude_b,
            longitude_b,
        )

        bearing = initial_bearing(
            latitude_a,
            longitude_a,
            latitude_b,
            longitude_b,
        )

        st.subheader("📊 Link Summary")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Link Distance",
            f"{distance_m / 1000:.3f} km",
        )

        c2.metric(
            "Azimuth A → B",
            f"{bearing:.2f}°",
        )

        c3.metric(
            "Wavelength",
            f"{wavelength_nm:.0f} nm",
        )

        c4.metric(
            "Beam Divergence",
            f"{beam_divergence:.2f}°",
        )

        # ----------------------------------------------------
        # Terrain retrieval
        # ----------------------------------------------------

        with st.spinner("Retrieving terrain elevation..."):
            terrain_df = get_terrain_profile(
                latitude_a,
                longitude_a,
                latitude_b,
                longitude_b,
                distance_m,
                sample_points,
            )

        if terrain_df["terrain_elevation_m"].isna().all():
            st.error("No terrain elevation data was returned.")
            st.stop()

        # ----------------------------------------------------
        # LOS analysis
        # ----------------------------------------------------

        result = analyze_los(
            terrain_df,
            height_a,
            height_b,
            wavelength_nm,
            fresnel_percentage,
        )
        # ----------------------------------------------------
        # Recommended Mounting Height Display
        # ----------------------------------------------------

        rec_height = calculate_min_mounting_heights(
            terrain_df, 
            distance_m, 
            clearance_margin_m=1.0 # 1 meter clearance safety margin
        )

        st.subheader("💡 Recommended Mounting Height")

        if height_a >= rec_height and height_b >= rec_height:
            st.success(
                f"Your current mounting heights ({height_a:.1f} m / {height_b:.1f} m) "
                f"are sufficient! Recommended minimum height for both sites is **{rec_height:.2f} m**."
            )
        else:
            st.warning(
                f"To achieve full terrain clearance with a 1m safety buffer, "
                f"increase mounting heights at both sites to at least **{rec_height:.2f} m**."
            )

        df = result["data"]

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        st.subheader("🛰️ Terrain / LOS Result")

        geometric_clearance = result["minimum_los_clearance"]
        fresnel_clearance = result["minimum_fresnel_clearance"]

        # Overall status
        los_status = "PASS" if geometric_clearance > 0 else "FAIL"
        fresnel_status = "PASS" if fresnel_clearance >= 0 else "WARNING"

        # Status cards
        r1, r2, r3 = st.columns(3)

        if los_status == "PASS":
            r1.success(
                f"### LOS: {los_status}\n"
                f"Minimum clearance: {geometric_clearance:.2f} m"
            )
        else:
            r1.error(
                f"### LOS: {los_status}\n"
                f"Minimum clearance: {geometric_clearance:.2f} m"
            )

        if fresnel_status == "PASS":
            r2.success(
                f"### FRESNEL: {fresnel_status}\n"
                f"Minimum clearance: {fresnel_clearance:.2f} m"
            )
        else:
            r2.warning(
                f"### FRESNEL: {fresnel_status}\n"
                f"Minimum clearance: {fresnel_clearance:.2f} m"
            )

        if los_status == "PASS" and fresnel_status == "PASS":
            r3.success("### PRELIMINARY\nFEASIBLE")
        elif los_status == "PASS":
            r3.warning("### PRELIMINARY\nREVIEW REQUIRED")
        else:
            r3.error("### PRELIMINARY\nNOT FEASIBLE")

        # ----------------------------------------------------
        # Site elevation information
        # ----------------------------------------------------

        st.subheader("📍 Site Elevations")

        e1, e2, e3, e4 = st.columns(4)

        e1.metric("Ground A", f"{result['ground_a']:.1f} m")
        e2.metric("Device A", f"{result['aperture_a']:.1f} m")
        e3.metric("Ground B", f"{result['ground_b']:.1f} m")
        e4.metric("Device B", f"{result['aperture_b']:.1f} m")

        # ----------------------------------------------------
        # Critical point
        # ----------------------------------------------------

        st.subheader("⚠️ Critical Terrain Point")

        critical = df.iloc[result["critical_los_index"]]

        st.write(
            f"""
**Distance from Site A:** {critical['distance_m']:.1f} m  
**Coordinates:** {critical['latitude']:.6f}, {critical['longitude']:.6f}  
**Terrain elevation:** {critical['terrain_elevation_m']:.1f} m  
**LOS elevation:** {critical['los_elevation_m']:.1f} m  
**Clearance:** {critical['geometric_clearance_m']:.2f} m
"""
        )

        # ----------------------------------------------------
        # INTERACTIVE MAP
        # ----------------------------------------------------

        st.subheader("🗺️ Link Path Map")

        map_lat = (latitude_a + latitude_b) / 2
        map_lon = (longitude_a + longitude_b) / 2

        map_fig = go.Figure()

        # A → B link line
        map_fig.add_trace(
            go.Scattermap(
                lat=[latitude_a, latitude_b],
                lon=[longitude_a, longitude_b],
                mode="lines",
                line=dict(width=4),
                name="LC LYNC Link"
            )
        )

        # Site A marker
        map_fig.add_trace(
            go.Scattermap(
                lat=[latitude_a],
                lon=[longitude_a],
                mode="markers+text",
                marker=dict(size=14),
                text=["SITE A"],
                textposition="top center",
                name="Site A"
            )
        )

        # Site B marker
        map_fig.add_trace(
            go.Scattermap(
                lat=[latitude_b],
                lon=[longitude_b],
                mode="markers+text",
                marker=dict(size=14),
                text=["SITE B"],
                textposition="top center",
                name="Site B"
            )
        )

        map_fig.update_layout(
            map=dict(
                style="open-street-map",
                center=dict(lat=map_lat, lon=map_lon),
                zoom=15
            ),
            height=550,
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h")
        )

        st.plotly_chart(map_fig, use_container_width=True)

        # ----------------------------------------------------
        # GOOGLE EARTH VERIFICATION
        # ----------------------------------------------------

        st.subheader("🌍 Google Earth Verification")

        st.write(
            "Enter the distance and heading measured in Google Earth "
            "to compare them with the calculator."
        )

        ge1, ge2 = st.columns(2)

        with ge1:
            google_earth_distance = st.number_input(
                "Google Earth distance (m)",
                min_value=0.0,
                value=274.43,
                step=0.01
            )

        with ge2:
            google_earth_heading = st.number_input(
                "Google Earth heading (°)",
                min_value=0.0,
                max_value=360.0,
                value=199.59,
                step=0.01
            )

        distance_difference = distance_m - google_earth_distance
        distance_difference_abs = abs(distance_difference)
        distance_difference_percent = (
            distance_difference_abs / google_earth_distance * 100
            if google_earth_distance > 0
            else 0
        )

        heading_difference = abs(bearing - google_earth_heading)

        if heading_difference > 180:
            heading_difference = 360 - heading_difference

        v1, v2, v3, v4 = st.columns(4)

        v1.metric("Tool Distance", f"{distance_m:.2f} m")
        v2.metric("Google Earth", f"{google_earth_distance:.2f} m")
        v3.metric("Distance Difference", f"{distance_difference_abs:.2f} m")
        v4.metric("Difference %", f"{distance_difference_percent:.2f}%")

        h1, h2 = st.columns(2)

        h1.metric("Tool Heading", f"{bearing:.2f}°")
        h2.metric("Heading Difference", f"{heading_difference:.2f}°")

        if distance_difference_percent <= 1:
            st.success("✓ Distance agrees with Google Earth within 1%.")
        elif distance_difference_percent <= 3:
            st.warning(
                "⚠ Distance differs by more than 1%. "
                "Check that both tools use exactly the same coordinates."
            )
        else:
            st.error(
                "✗ Significant distance difference. "
                "Verify Site A and Site B coordinates."
            )

        # ----------------------------------------------------
        # Terrain graph
        # ----------------------------------------------------

        st.subheader("⛰️ Terrain / Optical LOS Profile")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["distance_m"],
                y=df["terrain_elevation_m"],
                mode="lines",
                name="Terrain",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df["distance_m"],
                y=df["los_elevation_m"],
                mode="lines",
                name="Optical LOS",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df["distance_m"],
                y=df["los_elevation_m"] - df["required_fresnel_clearance_m"],
                mode="lines",
                name=f"LOS - {fresnel_percentage}% Fresnel",
            )
        )

        # Critical point
        fig.add_trace(
            go.Scatter(
                x=[critical["distance_m"]],
                y=[critical["terrain_elevation_m"]],
                mode="markers",
                marker=dict(size=12),
                name="Critical Point",
            )
        )

        fig.update_layout(
            title="LC LYNC Terrain / LOS / Fresnel Profile",
            xaxis_title="Distance from Site A (m)",
            yaxis_title="Elevation (m)",
            hovermode="x unified",
            height=550,
        )

        st.plotly_chart(fig, use_container_width=True)

        # ----------------------------------------------------
        # Link geometry
        # ----------------------------------------------------

        st.subheader("📐 Link Geometry")

        g1, g2, g3 = st.columns(3)

        g1.metric("Max 1st Fresnel Radius", f"{result['max_fresnel_radius']:.3f} m")
        g2.metric("Device A Elevation", f"{result['aperture_a']:.2f} m")
        g3.metric("Device B Elevation", f"{result['aperture_b']:.2f} m")

        # ----------------------------------------------------
        # Optical beam information
        # ----------------------------------------------------

        st.subheader("🔦 Optical Beam Estimate")

        divergence_rad = math.radians(beam_divergence)
        beam_diameter_m = 2 * distance_m * math.tan(divergence_rad / 2)

        ob1, ob2, ob3 = st.columns(3)

        ob1.metric("Optical Aperture", f"{optical_aperture:.1f} mm")
        ob2.metric("Estimated Beam Diameter", f"{beam_diameter_m:.2f} m")
        ob3.metric("Beam Divergence", f"{beam_divergence:.2f}°")

        # ----------------------------------------------------
        # CSV export
        # ----------------------------------------------------

        st.subheader("📥 Export Survey Data")

        csv_data = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download LOS Survey CSV",
            data=csv_data,
            file_name="LC_LYNC_LOS_Survey.csv",
            mime="text/csv",
        )

        # ----------------------------------------------------
        # Engineering limitation
        # ----------------------------------------------------

        st.warning(
            """
### ⚠️ Important Engineering Limitation

This calculator currently performs **terrain-based preliminary feasibility**.

The elevation dataset does NOT guarantee detection of:
- Trees
- Buildings
- Poles
- Cranes
- Towers
- Transmission lines
- Temporary structures
- Other narrow optical obstructions

For an 850 nm LC LYNC optical link, the final installation should therefore
include a physical/satellite site survey and confirmation of the actual optical
path.

**PASS means terrain-based preliminary feasibility, not final installation approval.**
"""
        )

    except requests.exceptions.RequestException as error:
        st.error("Elevation service could not be reached.")
        st.code(str(error))
        st.info("Please try the analysis again after a short interval.")

    except Exception as error:
        st.error("Analysis failed.")
        st.exception(error)


# ============================================================
# INITIAL SCREEN
# ============================================================

else:

    st.info(
        """
Enter the GPS coordinates and device mounting heights
in the sidebar, then click **ANALYZE LINK**.
"""
    )

    st.markdown(
        """
## What this version calculates

### 📍 Link
- Site A GPS
- Site B GPS
- Distance
- Azimuth

### ⛰️ Terrain
- Intermediate GPS points
- Terrain elevation
- Terrain profile
- Earth curvature

### 👁️ LOS
- Actual device aperture elevation
- Straight optical path
- Minimum terrain clearance
- Critical terrain point

### 📐 Fresnel
- 850 nm wavelength
- First Fresnel-zone radius
- Configurable clearance percentage

### 📡 Optical
- Beam divergence
- Optical aperture
- Approximate beam diameter

### 📥 Export
- Complete terrain/LOS CSV

---

### Recommended interpretation

**PASS**
Terrain-based LOS is clear and the selected Fresnel criterion is satisfied.

**WARNING**
Geometric LOS is clear, but the selected Fresnel criterion is not satisfied.

**FAIL**
Terrain intersects the optical LOS centerline.

**Important:** This is a preliminary planning tool. Trees, buildings,
poles and other narrow objects require separate verification.
"""
    )
