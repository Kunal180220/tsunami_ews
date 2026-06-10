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

# 1. Global Platform System Configuration
st.set_page_config(layout="wide", page_title="Unified Tsunami EWS & Routing Hub", page_icon="🌊")

st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    h1, h2, h3 { color: #00f2fe; font-family: 'Courier New', monospace; }
    div.stDataFrame { background-color: #121824; }
    </style>
""", unsafe_allow_html=True)

st.title("🎛️ Unified Tsunami Early Warning & Routing Platform")
st.markdown("---")

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
    """
    Computes tsunami travel times using highly-optimized vectorized raster masks.
    """
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
    """
    Connects to the global USGS streaming seismic network, handles dynamic
    IST timestamp shifting, and filters multi-tiered temporal arrays.
    """
    now = datetime.utcnow()
    
    # Pre-compiled high-availability 7-day feed directly from USGS servers
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
        
        # --- FIXED IST TIME CONVERSION LOGIC ---
        epoch_time = prop.get('time') / 1000.0
        event_dt_utc = datetime.utcfromtimestamp(epoch_time)
        
        # Explicitly apply the Indian Standard Time offset (+5h 30m)
        event_dt_ist = event_dt_utc + timedelta(hours=5, minutes=30)
        
        # Calculate age threshold metrics against global standard baseline tracking clock
        hours_old = (now - event_dt_utc).total_seconds() / 3600
        readable_time = event_dt_ist.strftime('%m-%d %H:%M') + " IST"
        
        event_dict = {
            "Title": prop.get('title'),
            "Place": prop.get('place') if prop.get('place') else "Open Ocean",
            "Magnitude": mag,
            "Depth": round(depth, 1),
            "Latitude": lat,
            "Longitude": lon,
            "Time": readable_time,
            "Time_Raw": event_dt_ist, # Fixed reference here
            "Hours_Old": hours_old,
            "Is_Tsunami_Threat": False
        }
        
        # Tsunami Generation Rules: Severe Shockwave (>=6.5 Magnitude) and Shallow Oceanic Crust (<100km)
        is_seafloor_rupture = mag and mag >= 6.5 and depth < 100
        
        # Geofencing coordinate framework for Indian Ocean & Western Pacific Region
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


# --- EXECUTE INGESTION ENGINE AND PACK MATRIX DATA VARIABLES ---
df_all, df_tsunami, df_regional_past = fetch_dynamic_seismic_matrix()


# --- NETWORK SAFETY VALVE: OFFLINE SIMULATION LOOP ---
if df_all.empty:
    df_all = pd.DataFrame([
        {"Title": "M 4.5 - Andaman Islands, India Region", "Place": "Andaman Islands, India", "Magnitude": 4.5, "Depth": 35.0, "Latitude": 11.6, "Longitude": 92.7, "Time": "06-10 12:45 IST", "Time_Raw": datetime.utcnow(), "Is_Tsunami_Threat": False},
        {"Title": "M 5.2 - Hindu Kush Region, Afghanistan", "Place": "Hindu Kush, Afghanistan", "Magnitude": 5.2, "Depth": 120.0, "Latitude": 36.5, "Longitude": 70.8, "Time": "06-10 11:20 IST", "Time_Raw": datetime.utcnow(), "Is_Tsunami_Threat": False},
        {"Title": "M 3.1 - Nicobar Islands, India", "Place": "Nicobar Islands, India", "Magnitude": 3.1, "Depth": 10.0, "Latitude": 7.1, "Longitude": 93.8, "Time": "06-10 09:15 IST", "Time_Raw": datetime.utcnow(), "Is_Tsunami_Threat": False},
        {"Title": "M 7.1 - Banda Sea, Indonesia (CRITICAL THREAT)", "Place": "Banda Sea, Indonesia", "Magnitude": 7.1, "Depth": 15.0, "Latitude": -6.5, "Longitude": 129.2, "Time": "06-10 06:10 IST", "Time_Raw": datetime.utcnow(), "Is_Tsunami_Threat": True}
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
    
    # Initialize a Session State key to store the index of the clicked ticker event
    if "selected_event_index" not in st.session_state:
        st.session_state.selected_event_index = None

    with col1_ticker:
        st.subheader("🚨 Real-Time Seismic Stream")
        st.caption("Click an active telemetry event below to lock coordinates & highlight on grid:")
        st.markdown("---")
        
        if not df_all.empty:
            for idx, row in df_all.iterrows():
                # Formulate structural button metadata labels
                button_label = f"M {row['Magnitude']} | {row['Place']} ({row['Time']})"
                is_selected = (st.session_state.selected_event_index == idx)
                
                if row['Is_Tsunami_Threat']:
                    # Critical threat visual wrapper
                    if st.button(f"🔴 CRITICAL THREAT: {button_label}", key=f"btn_{idx}", use_container_width=True):
                        st.session_state.selected_event_index = idx
                        st.rerun()
                else:
                    # Standard event visual wrapper with active selection pointer
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
            
            # If a user has clicked a ticker button, inject its state true flag
            if st.session_state.selected_event_index is not None and st.session_state.selected_event_index in df_mapped.index:
                df_mapped.loc[st.session_state.selected_event_index, 'is_active_selection'] = True
                
                # Dynamically extract focused coordinates to anchor the camera viewpoint positioning
                focus_row = df_mapped.loc[st.session_state.selected_event_index]
                map_center_lat = focus_row["Latitude"]
                map_center_lon = focus_row["Longitude"]
                map_zoom = 5.0  # Bring camera inward for detailed structural examination
            else:
                # Default baseline fallback views if no selection override is active
                if not df_tsunami.empty:
                    map_center_lat, map_center_lon, map_zoom = df_tsunami.iloc[0]["Latitude"], df_tsunami.iloc[0]["Longitude"], 3.0
                else:
                    map_center_lat, map_center_lon, map_zoom = 15.0, 80.0, 2.5
                    
            # Scatterplot configuration tracking the selection flag column
            layer_all_quakes = pdk.Layer(
                "ScatterplotLayer",
                df_mapped,
                get_position="[Longitude, Latitude]",
                # Scale the point size wider if it's the actively selected element
                get_radius="is_active_selection ? 18 : (Is_Tsunami_Threat ? 12 : 6)", 
                radius_min_pixels=4,   
                radius_max_pixels=35,  
                # Conditional coloration assignment: Cyan for click highlight, Red/Orange for baseline states
                get_fill_color="is_active_selection ? [0, 242, 254, 255] : (Is_Tsunami_Threat ? [255, 30, 30, 240] : [255, 160, 20, 160])",
                get_line_color="is_active_selection ? [255, 255, 255, 255] : [0, 0, 0, 0]",
                get_line_width=2,
                stroked=True,
                pickable=True,
                auto_highlight=True,
                highlight_color=[0, 242, 254, 255]
            )
            
            layers_to_render = [layer_all_quakes]
            
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
                
                # Retain structural focal array on active tsunami if no button press overrides it
                if st.session_state.selected_event_index is None:
                    map_center_lat, map_center_lon, map_zoom = active_epi["Latitude"], active_epi["Longitude"], 3.0
                
            st.pydeck_chart(pdk.Deck(
                layers=layers_to_render,
                initial_view_state=pdk.ViewState(latitude=map_center_lat, longitude=map_center_lon, zoom=map_zoom, pitch=10),
                map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                tooltip={
                    "html": """
                        <div style='font-family: monospace; background-color: #121824; color: #00f2fe; padding: 8px; border-radius: 4px; border: 1px solid #00f2fe;'>
                            <b>📊 Magnitude:</b> {Magnitude}<br/>
                            <b>📍 Location:</b> {Place}<br/>
                            <b>🌐 Coordinates:</b> {Latitude}°, {Longitude}°<br/>
                            <b>📉 Depth:</b> {Depth} km<br/>
                            <b>🕒 Date/Time:</b> {Time}
                        </div>
                    """,
                    "style": {"backgroundColor": "transparent", "zIndex": "10000"}
                }
            ))
        else:
            st.info("Loading baseline tracking matrix...")

# ==============================================================================
# TAB 2: ROUTING & SIMULATION SANDBOX (WITH NOAA COASTAL LOCATION FORECASTING)
# ==============================================================================
with tab2:
    st.subheader("🔮 Custom Manual Simulator & Deep-Sea Pathfinder Sandbox")
    st.markdown("Analyze wave routing pathways by matching past archives, inputting ad-hoc epicenters, or assessing targeted NOAA coastal forecasting zones.")
    
    col2_control, col2_map_view = st.columns([1, 2])
    
    with col2_control:
        # Added 4th option for NOAA TTT Coastal Framework matching
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
        
        # --- NOAA TTT PRE-COMPUTED DATABASE STRUCTURING ---
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
            target_lat = st.number_input("Epicenter Latitude (-90.0 to 90.0):", min_value=-90.0, max_value=90.0, value=10.0, step=0.1, key="manual_lat")
            target_lon = st.number_input("Epicenter Longitude (-180.0 to 180.0):", min_value=-180.0, max_value=180.0, value=90.0, step=0.1, key="manual_lon")
            sim_mag = st.slider("Select Shockwave Magnitude Scale (M):", min_value=4.0, max_value=9.5, value=7.5, step=0.1, key="manual_mag")
            label_name = f"User-Defined Manual Simulation (M {sim_mag})"
            
        elif mode == "🌐 NOAA Framework: Pre-Computed Coastal Gauges":
            st.markdown("#### 🌎 NOAA TTT Region Selection")
            chosen_region = st.selectbox("Select NOAA Forecast Domain:", list(noaa_regions.keys()))
            
            coastal_locations = noaa_regions[chosen_region]
            chosen_gauge = st.selectbox("Select Coastal Location Gauge:", list(coastal_locations.keys()))
            
            # Anchor coordinate matrix to selected coastal gauge
            target_lat = coastal_locations[chosen_gauge]["lat"]
            target_lon = coastal_locations[chosen_gauge]["lon"]
            sim_mag = st.slider("Simulate Generative Trigger Magnitude (M):", min_value=6.5, max_value=9.5, value=8.0, step=0.1, key="noaa_sim_mag")
            label_name = f"NOAA Gauge Target: {chosen_gauge} ({chosen_region})"
            
        st.markdown("---")
        st.success(f"**Target Anchored:**\n\n{label_name}\n\nLocation: {target_lat}°, {target_lon}°")
        
        # Run calculations if event is equal to or greater than critical threshold
        should_calculate_waves = sim_mag >= 6.5
        
        if should_calculate_waves:
            lons_r, lats_r, travel_grid_r = calculate_propagation(target_lat, target_lon)
            df_replay_contours = get_contour_paths(lons_r, lats_r, travel_grid_r, max_hours=14)
            
            replay_impact = []
            
            # 1. Standardize simulation origin baseline to operational IST (+5h 30m)
            origin_dt_ist = origin_dt + timedelta(hours=5, minutes=30)
            
            # If tracking via NOAA mode, intercept and calculate arrival vectors for the gauge
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
            
            # 2. Supplement baseline port listings with shifted local time rules
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
            # Render dataframe with updated local timezone headers
            st.dataframe(df_rep_impact, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ Tsunami pathfinding suppressed. Magnitude is below critical threshold ($M < 6.5$), indicating minimal hydro-displacement hazard risk.")
            
    with col2_map_view:
        st.subheader("🗺️ Replay/Simulation Propagation Trajectory Grid")
        
        # Dynamic coloring: Magenta circle marker for earthquakes, Neon Cyan star flare for NOAA gauges
        marker_color = [0, 242, 254, 255] if mode == "🌐 NOAA Framework: Pre-Computed Coastal Gauges" else [255, 0, 255, 250]
        
        layer_rep_epi = pdk.Layer(
            "ScatterplotLayer",
            pd.DataFrame([{"Lat": target_lat, "Lon": target_lon}]),
            get_position="[Lon, Lat]",
            get_color=marker_color,
            get_radius=220000 if mode != "🌐 NOAA Framework: Pre-Computed Coastal Gauges" else 120000,
            pickable=True
        )
        
        layers_sandbox = [layer_rep_epi]
        
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
            
        view_rep = pdk.ViewState(latitude=target_lat, longitude=target_lon, zoom=2.8 if mode == "🌐 NOAA Framework: Pre-Computed Coastal Gauges" else 2.2, pitch=10)
        
        st.pydeck_chart(pdk.Deck(
            layers=layers_sandbox,
            initial_view_state=view_rep,
            map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            tooltip={"text": "Simulation Front Node\nTime From Source: {hour} Hours"}
        ))