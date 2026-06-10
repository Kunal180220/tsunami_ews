# 🌊 Real-Time Tsunami Early Warning System (EWS)

An interactive, live-streaming seismic monitoring dashboard and wave propagation simulator. This platform tracks global earthquake data in real-time, instantly flags events capable of triggering tsunamis, and maps potential wave travel pathways across the ocean.

### 🖥️ Live Operations Center Overview
![Live Operations Room](./images/Live_Operations_Room.png)
---

## 🚀 Key Features

* **🛰️ Live Seismic Telemetry:** Connects directly to the live United States Geological Survey (USGS) feed, updating automatically every 30 seconds.
* **⚠️ Automated Threat Filtering:** Automatically isolates and flags critical oceanic earthquakes (Magnitude $\ge 6.5$ and Depth $< 100\text{ km}$) that possess the physical characteristics to displace seawater.
* **🕒 Localized Operational Clock:** Dynamically shifts all standard global data timestamps into Indian Standard Time (IST) for seamless localized monitoring.
* **🔮 Interactive Simulation Sandbox:** Allows operators to manually simulate custom deep-sea epicenters or test fixed coastal monitoring networks using a simulation framework modeled after NOAA's Tsunami Travel Time (TTT) specifications.
* **🎯 Bidirectional Workspace Sync:** Clicking any incoming event in the right-side live stream panel automatically focuses, zooms, and highlights that specific coordinate point on the map grid.

---

## 🔬 How It Works (The Methodology)

Instead of using massive supercomputers that take hours to calculate deep-ocean tidal dynamics, this platform uses optimized, real-time matrix math to calculate emergency travel-time projections instantly.

### 1. Ocean Wave Physics
Tsunami waves have massive wavelengths compared to the depth of the ocean. Because of this, they behave strictly as **shallow-water waves**. Their speed ($v$) across the sea depends entirely on the acceleration of gravity ($g$) and the depth of the water ($h$):

$$v = \sqrt{g \cdot h}$$

* Deep water = Waves travel incredibly fast (up to 800 km/h, the speed of a jet airliner).
* Shallow water near coasts = Waves slow down significantly but stack upward in height.

### 2. The Fast Marching Travel Engine
To track how waves move around island chains, deep trenches, and irregular coastlines, the app uses the **Fast Marching Method (FMM)** via the `scikit-fmm` computer engine. 

The algorithm treats the epicenter as a pebble dropped in water and solves the **Eikonal Equation** across a 2D grid map of the Earth:

$$|\nabla T| \cdot v = 1$$

This equation maps out the arrival time ($T$) for every coordinate on the map. Because the engine checks the actual local water depth data dynamically:
* Contour lines stretch far apart in deep ocean basins (signaling high-velocity propagation).
* Contour lines compress tightly near shallow coastal borders (signaling wave slowing and shoaling risks).

---

## 📂 Project Architecture

This application operates **100% in system memory (RAM)**. It maintains no heavy local databases, CSV logs, or image dependencies on your computer, making it completely portable and fast.

#### A. Archive Replay & Historical Anchor Profiles
![Historical Threat Replication](./images/Historical_Events.png)

#### B. Completely Manual Epicentral Controls
![Ad-hoc Generative Parameters](./images/Manual_Simulation.png)

#### C. NOAA TTT Pre-Computed Gauge Framework
![Targeted Coastal Forecast Nodes](./images/NOAA_Framework.png)

#### D. Recent Live Threat Replays
![Recent Regional Analysis](./images/Recent_Threats.png)

```text
tsunami_ews/
├── .gitignore          # Keeps your directory clean by ignoring local caches
├── app.py              # The main, self-contained Streamlit dashboard application
└── README.md           # This project documentation webpage