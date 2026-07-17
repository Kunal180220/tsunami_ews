import streamlit as st
import pandas as pd
import pydeck as pdk
import requests
import numpy as np
import skfmm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio import features
from rasterio.transform import from_bounds
import time

# 1. Global Platform System Configuration
st.set_page_config(layout="wide", page_title="Unified Tsunami EWS & Routing Hub", page_icon="🌊")

# Find your sidebar header/title, and paste right beneath it:
st.sidebar.title("📟 Live Operations Center")

# Dynamic operational layer switch visibility flags
show_faults = st.sidebar.checkbox("👁️ Display Tectonic Plate Fault Lines", value=True)
show_heatmap = st.sidebar.checkbox("👁️ Display Seismic Activity Heatmap", value=True)

st.sidebar.markdown("""
---
⚠️ **National Border Disclaimer:** *This application uses international map tiles provided by third-party open-source libraries (CARTO/OpenStreetMap). The borders shown do not imply the expression of any opinion whatsoever concerning the legal status of any country or territory. The developer recognizes the entire territory of Jammu, Kashmir, and Ladakh as an integral part of India in accordance with the official maps published by the Survey of India.*
""")

st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    h1, h2, h3 { color: #00f2fe; font-family: 'Courier New', monospace; }
    div.stDataFrame { background-color: #121824; }
    </style>
""", unsafe_allow_html=True)

st.title("🎛️ Unified Tsunami Early Warning & Routing Platform")
st.markdown("---")

# ==============================================================================
# ANIMATION CALCULATOR: Generate Heartbeat Waveform Scalars
# ==============================================================================
# Uses time-based trigonometry to create a smooth, continuous contraction/expansion pulse
current_timestamp = time.time()
# heartbeat formula produces an oscillation between ~0.7 and ~1.3
heartbeat_pulse = 1.0 + 0.3 * np.sin(current_timestamp * 2 * np.pi * 1.2) 

# ==============================================================================
# DATA CORE: Define the 3 prominent regional plate boundary/fault zone paths
# ==============================================================================
faults_data = [
    {
        "name": "Sunda Megathrust / Sumatra Subduction Zone",
        "reason": "The Indo-Australian Plate is actively subducting beneath the Burma/Sunda microplates at ~60mm/yr, generating massive megathrust slip failures.",
        "path": [[95.0, 6.0], [94.0, 2.0], [93.0, -2.0], [97.0, -6.0]],
        "color": [255, 69, 0, 180]  # Orange-Red
    },
    {
        "name": "Andaman Rift Valley / Transform Fault Complex",
        "reason": "Oblique tectonic shearing and sea-floor spreading along the Andaman sea basin edge creating transform strike-slip faults.",
        "path": [[92.0, 14.0], [93.5, 11.0], [95.0, 6.0]],
        "color": [255, 215, 0, 180]  # Gold
    },
    {
        "name": "Owen Fracture Zone / Arabian Sea Boundary",
        "reason": "A major dextral strike-slip transform fault separating the Arabian Plate from the Indo-Australian Plate.",
        "path": [[60.0, 20.0], [62.0, 15.0], [65.0, 10.0]],
        "color": [50, 205, 50, 180]  # Lime Green
    }
]
df_faults = pd.DataFrame(faults_data)

def find_nearest_fault_info(lat, lon):
    """Calculates the closest fault boundary to any coordinate point."""
    nearest_fault_name = "Unknown Plate Boundary"
    nearest_fault_reason = "Dynamic trigger outside primary regional network tracking layers."
    min_distance = float('inf')
    
    for fault in faults_data:
        for node in fault["path"]:
            dist = np.sqrt((node[1] - lat)**2 + (node[0] - lon)**2)
            if dist < min_distance:
                min_distance = dist
                nearest_fault_name = fault["name"]
                nearest_fault_reason = fault["reason"]
                
    return nearest_fault_name, nearest_fault_reason

# 2. Optimized High-Performance Wave Propagation Engine
@st.cache_data(ttl=3600)  # Cache the baseline land mask to keep performance lightning fast
def get_cached_land_mask(height, width):
    transform = from_bounds(-180, -90, 180, 90, width, height)
    try:
        world_url = "https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_50m_admin_0_countries.geojson"
        world = gpd.read_file(world_url)
        land_mask = features.rasterize(
            shapes=world.geometry,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            default_value=1,
            dtype=np.uint8
        ).astype(bool)
        return np.flipud(land_mask)
    except:
        return None

def calculate_propagation(epicenter_lat, epicenter_lon):
    """Computes tsunami travel times using highly-optimized vectorized raster masks."""
    height, width = 180, 360
    lats = np.linspace(-90, 90, height)
    lons = np.linspace(-180, 180, width)
    LON, LAT = np.meshgrid(lons, lats)
    
    bathymetry = -4200 + 3200 * np.sin(np.radians(LAT)) * np.cos(np.radians(LON))
    
    land_mask = get_cached_land_mask(height, width)
    if land_mask is not None:
        bathymetry[land_mask] = 500
    else:
        bathymetry[bathymetry > -150] = 500

    # Physics Core: Shallow Water Wave Speed v = sqrt(g * h)
    g = 9.81
    depth_profile = np.abs(bathymetry)
    depth_profile[bathymetry > 0] = 0.001  # Prevent division by zero on land
    
    speed_ms = np.sqrt(g * depth_profile)
    speed_deg_per_hr = (speed_ms * 3600) / 111000
    
    epi_y = (np.abs(lats - epicenter_lat)).argmin()
    epi_x = (np.abs(lons - epicenter_lon)).argmin()
    
    speed_field = np.array(speed_deg_per_hr)
    speed_field[bathymetry > 0] = 0.0001  # Land masses block waves naturally
    
    phi = np.ones_like(speed_field)
    phi[epi_y, epi_x] = -1
    
    try:
        travel_times_hr = skfmm.travel_time(phi, speed_field)
    except:
        travel_times_hr = np.sqrt((LAT - epicenter_lat)**2 + (LON - epicenter_lon)**2) / 7.2
        
    return lons, lats, travel_times_hr

# Extract map line contours out of the math engine matrix array
def get_contour_paths(lons, lats, travel_grid, max_hours=12):
    contours_data = []
    levels = np.arange(1, max_hours + 1, 1)
    
    fig, ax = plt.subplots()
    cs = ax.contour(lons, lats, travel_grid, levels=levels)
    
    paths = cs.get_paths()
    for i, path in enumerate(paths):
        level_hour = levels[i] if i < len(levels) else max_hours
        for polygon in path.to_polygons():
            path_coordinates = polygon.tolist()
            if len(path_coordinates) > 1:
                contours_data.append({
                    "path": path_coordinates,
                    "hour": int(level_hour),
                    "color": [0, 242, 254, 220] if level_hour % 2 == 0 else [0, 110, 255, 140]
                })
    plt.close(fig)
    return pd.DataFrame(contours_data)

# ==============================================================================
# 3. DYNAMIC MULTI-TIERED DATA INGESTION ENGINE & IST PROCESSING MATRIX
# ==============================================================================
@st.cache_data(ttl=30)
def fetch_dynamic_seismic_matrix():
    """Connects to the global USGS streaming seismic network, handles dynamic IST timestamp shifting, and filters multi-tiered temporal arrays."""
    now = datetime.utcnow()
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_week.geojson"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
    except:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    live_stream_24h = []      
    regional_past_3days = []  
    active_tsunami_threats = [] 
    
    for feature in data['features']:
        prop = feature['properties']
        geom = feature['geometry']
        mag = prop.get('mag')
        depth = geom['coordinates'][2]
        lon = geom['coordinates'][0]
        lat = geom['coordinates'][1]
        
        epoch_time = prop.get('time') / 1000.0
        event_dt_utc = datetime.utcfromtimestamp(epoch_time)
        event_dt_ist = event_dt_utc + timedelta(hours=5, minutes=30)
        
        hours_old = (now - event_dt_utc).total_seconds() / 3600
        readable_time = event_dt_ist.strftime('%m-%d %H:%M') + " IST"
        
        # Inject tectonic reasons dynamically for the event based on closest fault node
        fault_name, fault_reason = find_nearest_fault_info(lat, lon)
        
        event_dict = {
            "Title": prop.get('title'),
            "Place": prop.get('place') if prop.get('place') else "Open Ocean",
            "Magnitude": mag,
            "Depth": round(depth, 1),
            "Latitude": lat,
            "Longitude": lon,
            "Time": readable_time,
            "Time_Raw": event_dt_ist, 
            "Hours_Old": hours_old,
            "Is_Tsunami_Threat": False,
            "Fault": fault_name,
            "Reason": fault_reason
        }
        
        is_seafloor_rupture = mag and mag >= 6.5 and depth < 100
        is_in_monitored_zone = (30 <= lon <= 180) and (-50 <= lat <= 60)

        if is_seafloor_rupture:
            event_dict["Is_Tsunami_Threat"] = True
            if hours_old <= 24:
                active_tsunami_threats.append(event_dict)
            elif 24 < hours_old <= 72 and is_in_monitored_zone:
                regional_past_3days.append(event_dict)
                
        if hours_old <= 24:
            live_stream_24h.append(event_dict)
        
    return pd.DataFrame(live_stream_24h), pd.DataFrame(active_tsunami_threats), pd.DataFrame(regional_past_3days)

df_all, df_tsunami, df_regional_past = fetch_dynamic_seismic_matrix()

if df_all.empty:
    df_all = pd.DataFrame([
        {"Title": "M 4.5 - Andaman Islands, India Region", "Place": "Andaman Islands, India", "Magnitude": 4.5, "Depth": 35.0, "Latitude": 11.6, "Longitude": 92.7, "Time": "06-10 12:45 IST", "Time_Raw": datetime.utcnow(), "Is_Tsunami_Threat": False, "Fault": faults_data[1]["name"], "Reason": faults_data[1]["reason"]},
        {"Title": "M 5.2 - Hindu Kush Region, Afghanistan", "Place": "Hindu Kush, Afghanistan", "Magnitude": 5.2, "Depth": 120.0, "Latitude": 36.5, "Longitude": 70.8, "Time": "06-10 11:20 IST", "Time_Raw": datetime.utcnow(), "Is_Tsunami_Threat": False, "Fault": "Eurasian Plate Internal Fracture", "Reason": "Deep continental lithospheric compression."},
        {"Title": "M 3.1 - Nicobar Islands, India", "Place": "Nicobar Islands, India", "Magnitude": 3.1, "Depth": 10.0, "Latitude": 7.1, "Longitude": 93.8, "Time": "06-10 09:15 IST", "Time_Raw": datetime.utcnow(), "Is_Tsunami_Threat": False, "Fault": faults_data[0]["name"], "Reason": faults_data[0]["reason"]},
        {"Title": "M 7.1 - Banda Sea, Indonesia (CRITICAL THREAT)", "Place": "Banda Sea, Indonesia", "Magnitude": 7.1, "Depth": 15.0, "Latitude": -6.5, "Longitude": 129.2, "Time": "06-10 06:10 IST", "Time_Raw": datetime.utcnow(), "Is_Tsunami_Threat": True, "Fault": faults_data[0]["name"], "Reason": faults_data[0]["reason"]}
    ])
    df_tsunami = df_all[df_all["Is_Tsunami_Threat"] == True]
    df_regional_past = pd.DataFrame()

# 4. Target Watchlist
cities = pd.DataFrame([
    {"City": "Port Blair, Andaman Islands", "Lat": 11.62, "Lon": 92.73},
    {"City": "Chennai, India", "Lat": 13.08, "Lon": 80.27},
    {"City": "Mumbai, India", "Lat": 18.92, "Lon": 72.82},
    {"City": "Colombo, Sri Lanka", "Lat": 6.92, "Lon": 79.86},
    {"City": "Male, Maldives", "Lat": 4.17, "Lon": 73.51},
    {"City": "Tokyo, Japan", "Lat": 35.67, "Lon": 139.65},
    {"City": "Manila, Philippines", "Lat": 14.59, "Lon": 120.98},
    {"City": "Honolulu, Hawaii", "Lat": 21.30, "Lon": -157.85}
])

# 5. Master Navigation Interfaces
tab1, tab2 = st.tabs(["🖥️ LIVE OPERATIONS ROOM", "🗺️ HYDRODYNAMIC ROUTING & SIMULATION SANDBOX"])

# ==========================================
# TAB 1: LIVE OPERATIONS ROOM
# ==============================================
with tab1:
    col1_map, col1_ticker = st.columns([2, 1])
    if "selected_event_index" not in st.session_state:
        st.session_state.selected_event_index = None

    with col1_ticker:
        st.subheader("🚨 Real-Time Seismic Stream")
        st.caption("Click an active telemetry event below to lock coordinates & highlight on grid:")
        st.markdown("---")
        
        if not df_all.empty:
            for idx, row in df_all.iterrows():
                button_label = f"M {row['Magnitude']} | {row['Place']} ({row['Time']})"
                is_selected = (st.session_state.selected_event_index == idx)
                
                if row['Is_Tsunami_Threat']:
                    if st.button(f"🔴 CRITICAL THREAT: {button_label}", key=f"btn_{idx}", use_container_width=True):
                        st.session_state.selected_event_index = idx
                        st.rerun()
                else:
                    prefix = "▶️ " if is_selected else "🔸 "
                    if st.button(f"{prefix}{button_label}", key=f"btn_{idx}", use_container_width=True):
                        st.session_state.selected_event_index = idx
                        st.rerun()
        else:
            st.info("Awaiting next live telemetry ping from sensors...")
            
    with col1_map:
        st.subheader("📍 Real-Time Global Tactical Grid")
        if not df_all.empty:
            df_mapped = df_all.copy()
            df_mapped['is_active_selection'] = False
            
            # Isolate fresh items occurring within the last 2 hours
            df_mapped['is_latest'] = df_mapped['Hours_Old'] <= 2.0
            
            if st.session_state.selected_event_index is not None and st.session_state.selected_event_index in df_mapped.index:
                df_mapped.loc[st.session_state.selected_event_index, 'is_active_selection'] = True
                focus_row = df_mapped.loc[st.session_state.selected_event_index]
                map_center_lat = focus_row["Latitude"]
                map_center_lon = focus_row["Longitude"]
                map_zoom = 5.0  
            else:
                if not df_tsunami.empty:
                    map_center_lat, map_center_lon, map_zoom = df_tsunami.iloc[0]["Latitude"], df_tsunami.iloc[0]["Longitude"], 3.0
                else:
                    map_center_lat, map_center_lon, map_zoom = 15.0, 80.0, 2.5
                    
            layer_all_quakes = pdk.Layer(
                "ScatterplotLayer",
                df_mapped,
                get_position="[Longitude, Latitude]",
                get_radius="is_active_selection ? 180000 : (Is_Tsunami_Threat ? 120000 : 60000)", 
                radius_min_pixels=6,   
                radius_max_pixels=35,  
                get_fill_color="is_active_selection ? [0, 242, 254, 255] : (Is_Tsunami_Threat ? [255, 30, 30, 240] : [255, 160, 20, 160])",
                get_line_color="is_active_selection ? [255, 255, 255, 255] : [0, 0, 0, 0]",
                get_line_width=2,
                stroked=True,
                pickable=True,
                auto_highlight=True,
                highlight_color=[0, 242, 254, 255]
            )
            
            # Layer A1.3: Animated Outer Heartbeat Radar Ring
            # Evaluates the heartbeat expression variables to stretch metrics dynamically
            layer_latest_radar_ambient = pdk.Layer(
                "ScatterplotLayer",
                df_mapped[df_mapped['is_latest'] == True],
                get_position="[Longitude, Latitude]",
                get_radius=int(380000 * heartbeat_pulse),   # Pulsates the scale size factor
                radius_min_pixels=int(20 * heartbeat_pulse),
                radius_max_pixels=int(85 * heartbeat_pulse),
                get_fill_color=[0, 242, 254, int(25 / heartbeat_pulse)], # Fades out transparency as it expands
                get_line_color=[0, 242, 254, 100],
                get_line_width=1.5,
                stroked=True,
                filled=True,
                pickable=False
            )

            # Layer A1.6: Tight Inner Counter-Pulse Accent
            layer_latest_radar_core = pdk.Layer(
                "ScatterplotLayer",
                df_mapped[df_mapped['is_latest'] == True],
                get_position="[Longitude, Latitude]",
                get_radius=int(160000 * (2.0 - heartbeat_pulse)), # Inverts vector contraction direction
                radius_min_pixels=10,
                radius_max_pixels=45,
                get_fill_color=[255, 255, 255, 0],
                get_line_color=[255, 255, 255, int(200 * (heartbeat_pulse - 0.5))],
                get_line_width=3.5,
                stroked=True,
                filled=False,
                pickable=False
            )
            
            # Layer A2: Fault Lines Vectorization Network
            layer_fault_lines = pdk.Layer(
                "PathLayer",
                df_faults,
                get_path="path",
                get_color="color",
                width_scale=5,
                width_min_pixels=3,
                pickable=False
            )
            
            # Layer A3: Heatmap Intensity Engine
            layer_seismic_heatmap = pdk.Layer(
                "HeatmapLayer",
                df_mapped,
                get_position="[Longitude, Latitude]",
                get_weight="Magnitude",
                radius_pixels=65,
                intensity=1.8,
                threshold=0.03,
                opacity=0.7
            )
            
            layers_to_render = []
            
            if show_heatmap:
                layers_to_render.append(layer_seismic_heatmap)
            if show_faults:
                layers_to_render.append(layer_fault_lines)
                
            layers_to_render.append(layer_latest_radar_ambient)
            layers_to_render.append(layer_latest_radar_core)
            layers_to_render.append(layer_all_quakes)
            
            if not df_tsunami.empty:
                st.warning("⚠️ CRITICAL UNDERWATER EVENT RECOGNIZED. RAMPING UP WAVE PROPAGATION MODEL.")
                active_epi = df_tsunami.iloc[0]
                lons, lats, travel_grid = calculate_propagation(active_epi["Latitude"], active_epi["Longitude"])
                df_live_contours = get_contour_paths(lons, lats, travel_grid, max_hours=12)
                
                layer_live_contours = pdk.Layer(
                    "PathLayer",
                    df_live_contours,
                    get_path="path",
                    get_color="color",
                    width_scale=3,
                    width_min_pixels=2
                )
                layers_to_render.append(layer_live_contours)
                
                if st.session_state.selected_event_index is None:
                    map_center_lat, map_center_lon, map_zoom = active_epi["Latitude"], active_epi["Longitude"], 3.0
                
            st.pydeck_chart(pdk.Deck(
                layers=layers_to_render,
                initial_view_state=pdk.ViewState(latitude=map_center_lat, longitude=map_center_lon, zoom=map_zoom, pitch=10),
                map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                tooltip={
                    "html": """
                        <div style='font-family: monospace; background-color: #121824; color: #00f2fe; padding: 8px; border-radius: 4px; border: 1px solid #00f2fe; max-width: 300px;'>
                            <b>📊 Magnitude:</b> {Magnitude}<br/>
                            <b>📍 Location:</b> {Place}<br/>
                            <b>🌐 Coordinates:</b> {Latitude}°, {Longitude}°<br/>
                            <b>📉 Depth:</b> {Depth} km<br/>
                            <b>🛡️ Proximity Fault:</b> {Fault}<br/>
                            <b>💥 Trigger Reason:</b> {Reason}<br/>
                            <b>🕒 Time:</b> {Time}
                        </div>
                    """,
                    "style": {"backgroundColor": "transparent", "zIndex": "10000"}
                }
            ), height=650)
        else:
            st.info("Loading baseline tracking matrix...")

# ==============================================================================
# TAB 2: ROUTING & SIMULATION SANDBOX
# ==============================================================================
with tab2:
    st.subheader("🔮 Custom Manual Simulator & Deep-Sea Pathfinder Sandbox")
    st.markdown("Analyze wave routing pathways by matching past archives, inputting ad-hoc epicenters, or assessing targeted NOAA coastal forecasting zones.")
    
    col2_control, col2_map_view = st.columns([1, 2])
    with col2_control:
        mode = st.radio(
            "Choose Analysis Source Data Node:", 
            [
                "Replay Recent Live Threats", 
                "Simulate/Replay Historical Anchor Points", 
                "⚠️ Completely Manual Simulator Control",
                "🌐 NOAA Framework: Pre-Computed Coastal Gauges"
            ]
        )
        
        target_lat, target_lon, label_name, sim_mag = 0.0, 0.0, "", 7.5
        origin_dt = datetime.utcnow()
        
        noaa_regions = {
            "Atlantic & Caribbean": {
                "San Juan, Puerto Rico": {"lat": 18.46, "lon": -66.11},
                "Miami, Florida, USA": {"lat": 25.76, "lon": -80.19},
                "St. John's, Newfoundland": {"lat": 47.56, "lon": -52.71},
                "Bermuda (St. George's)": {"lat": 32.37, "lon": -64.67},
                "Charleston, South Carolina": {"lat": 32.77, "lon": -79.93}
            },
            "Indian Ocean & West Pacific": {
                "Port Blair, Andaman Islands": {"lat": 11.67, "lon": 92.74},
                "Colombo, Sri Lanka": {"lat": 6.92, "lon": 79.86},
                "Phuket, Thailand": {"lat": 7.88, "lon": 98.39},
                "Banda Aceh, Indonesia": {"lat": 5.54, "lon": 95.32},
                "Padang, Sumatra, Indonesia": {"lat": -0.94, "lon": 100.35},
                "Dili, Timor-Leste": {"lat": -8.55, "lon": 125.56},
                "Manila, Philippines": {"lat": 14.59, "lon": 120.98}
            },
            "Pacific Basin": {
                "Honolulu, Hawaii, USA": {"lat": 21.30, "lon": -157.85},
                "Guam (Apra Harbor)": {"lat": 13.44, "lon": 144.69},
                "Pago Pago, American Samoa": {"lat": -14.27, "lon": -170.70},
                "Valparaiso, Chile": {"lat": -33.04, "lon": -71.61},
                "Seward, Alaska, USA": {"lat": 60.10, "lon": -149.44},
                "Tokyo Bay, Japan": {"lat": 35.53, "lon": 139.78}
            }
        }

        if mode == "Replay Recent Live Threats":
            all_available_vectors = []
            if not df_tsunami.empty:
                all_available_vectors.extend(df_tsunami["Title"].tolist())
            if not df_regional_past.empty:
                all_available_vectors.extend(df_regional_past["Title"].tolist())
                
            if all_available_vectors:
                selected_choice = st.selectbox("Select Active or Recent Regional Event:", all_available_vectors)
                if not df_tsunami.empty and selected_choice in df_tsunami["Title"].values:
                    matched_row = df_tsunami[df_tsunami["Title"] == selected_choice].iloc[0]
                else:
                    matched_row = df_regional_past[df_regional_past["Title"] == selected_choice].iloc[0]
                target_lat = matched_row["Latitude"]
                target_lon = matched_row["Longitude"]
                label_name = matched_row["Title"]
                sim_mag = matched_row["Magnitude"]
                origin_dt = matched_row["Time_Raw"]
            else:
                st.info("No major regional threats found within 3 days. Defaulting to Sumatra anchor profile.")
                target_lat, target_lon, label_name, sim_mag = 3.31, 95.85, "2004 Sumatra-Andaman Archive", 9.1
                
        elif mode == "Simulate/Replay Historical Anchor Points":
            historical_db = {
                "2026 Mindanao, Philippines Trench Rupture (M 8.2)": {"lat": 5.59, "lon": 125.05, "mag": 8.2},
                "2004 Sumatra-Andaman Mega-Displacement (M 9.1)": {"lat": 3.31, "lon": 95.85, "mag": 9.1},
                "2011 Tohoku, Japan Structural Inundation (M 9.0)": {"lat": 38.32, "lon": 142.36, "mag": 9.0},
                "1960 Valdivia, Chile Great Rupture (M 9.5)": {"lat": -38.29, "lon": -73.05, "mag": 9.5}
            }
            hist_choice = st.selectbox("Select Historic Data Signature Archive:", list(historical_db.keys()))
            target_lat = historical_db[hist_choice]["lat"]
            target_lon = historical_db[hist_choice]["lon"]
            sim_mag = historical_db[hist_choice]["mag"]
            label_name = hist_choice
            
        elif mode == "⚠️ Completely Manual Simulator Control":
            st.markdown("#### 🛠️ Simulator Control Inputs")
            target_lat = st.number_input("Epicenter Latitude (-90.0 to 90.0):", min_value=-90.0, max_value=90.0, value=16.59, step=0.1, key="manual_lat")
            target_lon = st.number_input("Epicenter Longitude (-180.0 to 180.0):", min_value=-180.0, max_value=180.0, value=76.16, step=0.1, key="manual_lon")
            sim_mag = st.slider("Select Shockwave Magnitude Scale (M):", min_value=4.0, max_value=9.5, value=6.5, step=0.1, key="manual_mag")
            label_name = f"User-Defined Manual Simulation (M {sim_mag})"
            
        elif mode == "🌐 NOAA Framework: Pre-Computed Coastal Gauges":
            st.markdown("#### 🌎 NOAA TTT Region Selection")
            chosen_region = st.selectbox("Select NOAA Forecast Domain:", list(noaa_regions.keys()))
            
            coastal_locations = noaa_regions[chosen_region]
            chosen_gauge = st.selectbox("Select Coastal Location Gauge:", list(coastal_locations.keys()))
            
            target_lat = coastal_locations[chosen_gauge]["lat"]
            target_lon = coastal_locations[chosen_gauge]["lon"]
            sim_mag = st.slider("Simulate Generative Trigger Magnitude (M):", min_value=6.5, max_value=9.5, value=8.0, step=0.1, key="noaa_sim_mag")
            label_name = f"NOAA Gauge Target: {chosen_gauge} ({chosen_region})"
            
        st.markdown("---")
        st.success(f"**Target Anchored:**\n\n{label_name}\n\nLocation: {target_lat}°, {target_lon}°")
        
        sandbox_fault, sandbox_reason = find_nearest_fault_info(target_lat, target_lon)
        
        should_calculate_waves = sim_mag >= 6.5
        if should_calculate_waves:
            lons_r, lats_r, travel_grid_r = calculate_propagation(target_lat, target_lon)
            df_replay_contours = get_contour_paths(lons_r, lats_r, travel_grid_r, max_hours=14)
            replay_impact = []
            
            origin_dt_ist = origin_dt + timedelta(hours=5, minutes=30)
            if mode == "🌐 NOAA Framework: Pre-Computed Coastal Gauges":
                y_idx = (np.abs(lats_r - target_lat)).argmin()
                x_idx = (np.abs(lons_r - target_lon)).argmin()
                t_hr = travel_grid_r[y_idx, x_idx]
                
                lead_m = int(t_hr * 60) if t_hr > 0 else 0
                arrival_time_est = origin_dt_ist + timedelta(hours=float(t_hr))
                
                replay_impact.append({
                    "Target Station/Port": f"⭐ GAUGE: {chosen_gauge}",
                    "Travel Route Duration": f"{lead_m // 60}h {lead_m % 60}m" if lead_m > 0 else "Instant/Epicentral",
                    "Est. Arrival Time": arrival_time_est.strftime('%H:%M:%S') + " IST"
                })
            
            for _, city in cities.iterrows():
                y_idx = (np.abs(lats_r - city["Lat"])).argmin()
                x_idx = (np.abs(lons_r - city["Lon"])).argmin()
                t_hr = travel_grid_r[y_idx, x_idx]
                
                if 0 < t_hr < 24:
                    lead_m = int(t_hr * 60)
                    arrival_time_est = origin_dt_ist + timedelta(hours=float(t_hr))
                    replay_impact.append({
                        "Target Station/Port": city["City"],
                        "Travel Route Duration": f"{lead_m // 60}h {lead_m % 60}m",
                        "Est. Arrival Time": arrival_time_est.strftime('%H:%M:%S') + " IST"
                    })
            
            df_rep_impact = pd.DataFrame(replay_impact)
            st.subheader("📊 Intersected Arrival Log")
            st.dataframe(df_rep_impact, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ Tsunami pathfinding suppressed. Magnitude is below critical threshold ($M < 6.5$), indicating minimal hydro-displacement hazard risk.")
            
    with col2_map_view:
        st.subheader("🗺️ Replay/Simulation Propagation Trajectory Grid")
        
        marker_color = [0, 242, 254, 255] if mode == "🌐 NOAA Framework: Pre-Computed Coastal Gauges" else [255, 0, 255, 200]
        
        point_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([target_lon], [target_lat]),
            crs="EPSG:4326"
        )
        
        buffer_distance = 150000 if mode == "🌐 NOAA Framework: Pre-Computed Coastal Gauges" else 250000
        buffer_gdf = point_gdf.to_crs(epsg=3857).buffer(buffer_distance).to_crs(epsg=4326)
        
        df_sandbox_point = pd.DataFrame([{
            "Lat": target_lat,
            "Lon": target_lon,
            "Mag": sim_mag,
            "Fault": sandbox_fault,
            "Reason": sandbox_reason
        }])
        
        layer_rep_epi_buffer = pdk.Layer(
            "GeoJsonLayer",
            buffer_gdf.__geo_interface__,
            get_fill_color=marker_color + [60],  
            filled=True,
            stroked=False,
            pickable=False
        )
        
        layer_rep_epi_core = pdk.Layer(
            "ScatterplotLayer",
            df_sandbox_point,
            get_position="[Lon, Lat]",
            get_radius=80000,
            get_fill_color=marker_color,
            pickable=True
        )
        
        layer_sandbox_faults = pdk.Layer(
            "PathLayer",
            df_faults,
            get_path="path",
            get_color="color",
            width_scale=5,
            width_min_pixels=3,
            pickable=False
        )
        
        layers_sandbox = [layer_rep_epi_buffer, layer_rep_epi_core]
        
        if show_faults:
            layers_sandbox.insert(0, layer_sandbox_faults)
        
        if should_calculate_waves and 'df_replay_contours' in locals() and not df_replay_contours.empty:
            layer_rep_contours = pdk.Layer(
                "PathLayer",
                df_replay_contours,
                get_path="path",
                get_color="color",
                width_scale=3,
                width_min_pixels=2,
                pickable=True
            )
            layers_sandbox.append(layer_rep_contours)
            
        view_rep = pdk.ViewState(latitude=target_lat, longitude=target_lon, zoom=4.5 if mode == "🌐 NOAA Framework: Pre-Computed Coastal Gauges" else 3.8, pitch=10)
        
        st.pydeck_chart(pdk.Deck(
            layers=layers_sandbox,
            initial_view_state=view_rep,
            map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            tooltip={
                "html": """
                    <div style='font-family: monospace; background-color: #121824; color: #00f2fe; padding: 8px; border-radius: 4px; border: 1px solid #00f2fe; max-width: 300px;'>
                        <b>🔮 Simulation Node Trigger</b><br/>
                        <b>Coordinates:</b> {Lat}°, {Lon}°<br/>
                        <b>Assigned Magnitude:</b> M{Mag}<br/>
                        <b>Nearest Boundary:</b> {Fault}<br/>
                        <b>Tectonic Mechanism:</b> {Reason}
                    </div>
                """,
                "style": {"backgroundColor": "transparent", "zIndex": "10000"}
            }
        ), height=650)

# ==============================================================================
# 6. HIGH-FREQUENCY RADAR PULSE LOOP CONTROL
# ==============================================================================
# Forces Streamlit to instantly re-render every 150ms to drive the heartbeat fluidly
time.sleep(0.15)
st.rerun()
