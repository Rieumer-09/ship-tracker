# ship_tracker.py
import streamlit as st
import math, time
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ---------------- utilities ----------------
def deg2rad(d): return d * math.pi / 180.0
def rad2deg(r): return (r * 180.0 / math.pi) % 360.0
def wrap_pi(a): return (a + math.pi) % (2*math.pi) - math.pi
def tn_bearing_from_dxdy(dx, dy):
    ang = math.degrees(math.atan2(dx, dy))
    return (ang + 360.0) % 360.0

# ---------------- EKF (bearing-only) ----------------
class BearingOnlyEKF:
    """
    State x = [px, py, vx, vy]  (nm, nm, nm/min, nm/min)
    """
    def __init__(self, px, py, vx, vy):
        self.x = np.array([px, py, vx, vy], dtype=float)
        # covariance: position (nm^2), vel ((nm/min)^2)
        self.P = np.diag([25.0, 25.0, 0.1, 0.1])
        self.q = 1e-4
        self.R = (deg2rad(2.0))**2
        self.t_min = 0.0

    def predict(self, dt_min):
        F = np.array([[1,0,dt_min,0],
                      [0,1,0,dt_min],
                      [0,0,1,0],
                      [0,0,0,1]], dtype=float)
        self.x = F @ self.x
        # process noise for CV
        dt = dt_min
        G = np.array([[0.5*dt*dt, 0],
                      [0, 0.5*dt*dt],
                      [dt, 0],
                      [0, dt]])
        Q = G @ (self.q * np.eye(2)) @ G.T
        self.P = F @ self.P @ F.T + Q

    def update_bearing(self, bearing_rad, ox, oy):
        px, py, vx, vy = self.x
        dx = px - ox
        dy = py - oy
        r2 = dx*dx + dy*dy
        if r2 < 1e-9:
            return
        z_pred = math.atan2(dx, dy)
        y = wrap_pi(bearing_rad - z_pred)
        H = np.array([[ dy/r2, -dx/r2, 0.0, 0.0 ]], dtype=float)
        S = H @ self.P @ H.T + self.R
        K = (self.P @ H.T) / S
        self.x = self.x + (K.flatten() * y)
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P

    def estimate(self):
        px, py, vx, vy = self.x
        course = rad2deg(math.atan2(vx, vy))   # 0 = North, 90 = East
        speed_kn = math.hypot(vx, vy) * 60.0
        return px, py, vx, vy, course, speed_kn

# ---------------- safe session initialization ----------------
_defaults = {
    "own": {"course_deg": 90.0, "speed_kn": 12.0, "x": 0.0, "y": 0.0},
    "target_prior": {"bearing_deg": 0.0, "range_nm": 10.0, "speed_kn": 8.0, "geometry": "Opening"},
    "ekf": None,
    "log": [],                # each entry: dict with display + numeric fields for plotting
    "last_t_min": 0.0,
    "timer_running": False,
    "start_time": None,
    "initialized": False,     # EKF seeded when user clicks Initialize Target
    "first_bearing": None,
    "freeze": {"range": False, "range_mode": "Anchor", "speed": False, "course": False},
    "frozen_vals": {"range": None, "speed": None, "course": None},
    "plot_placeholder": None
}
for k,v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
# placeholder for radar plot
if st.session_state["plot_placeholder"] is None:
    st.session_state["plot_placeholder"] = st.empty()

# ---------------- helpers ----------------
def own_step(dt_min):
    """Advance ownship position by dt_min minutes using current own course/speed."""
    v_nm_min = st.session_state.own["speed_kn"] / 60.0
    cr = deg2rad(st.session_state.own["course_deg"])
    st.session_state.own["x"] += v_nm_min * math.sin(cr) * dt_min
    st.session_state.own["y"] += v_nm_min * math.cos(cr) * dt_min

def init_ekf_from_priors():
    """Seed EKF state from target_prior and current ownship pos. Does NOT touch timer."""
    prior = st.session_state.target_prior
    own = st.session_state.own
    B0 = deg2rad(prior["bearing_deg"])
    R0 = prior["range_nm"]
    dx = R0 * math.sin(B0)
    dy = R0 * math.cos(B0)
    px0 = own["x"] + dx
    py0 = own["y"] + dy
    v_nm_min = prior["speed_kn"]/60.0
    if prior["geometry"] == "Opening":
        cr = B0
    else:
        cr = (B0 + math.pi) % (2*math.pi)
    vx0 = v_nm_min * math.sin(cr)
    vy0 = v_nm_min * math.cos(cr)
    st.session_state.ekf = BearingOnlyEKF(px0, py0, vx0, vy0)
    st.session_state.initialized = True
    st.session_state.log = []
    st.session_state.last_t_min = 0.0
    st.session_state.first_bearing = None

def enforce_freeze_post_update():
    """Apply freezes to EKF state/covariance after an update; supports 'Anchor' mode for range."""
    ekf = st.session_state.ekf
    if ekf is None:
        return
    f = st.session_state.freeze
    fv = st.session_state.frozen_vals
    own = st.session_state.own

    # Range freeze modes:
    # - Lock: pin px,py to exactly the frozen radial distance (hard freeze)
    # - Anchor: set initial EKF range (if requested) but allow EKF to update range after that
    if f["range"] and (fv["range"] is not None):
        if f.get("range_mode", "Anchor") == "Lock":
            # compute direction own->target and place at exact radius
            px, py, vx, vy, crs, spd = ekf.estimate()
            dx = px - own["x"]
            dy = py - own["y"]
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                bearing = deg2rad(st.session_state.target_prior["bearing_deg"])
            else:
                bearing = math.atan2(dx, dy)
            ekf.x[0] = own["x"] + fv["range"] * math.sin(bearing)
            ekf.x[1] = own["y"] + fv["range"] * math.cos(bearing)
            ekf.P[0,0] = 1e-6; ekf.P[1,1] = 1e-6
        else:
            # Anchor mode: if EKF hasn't yet had many updates, we can set initial px,py
            # but we do NOT set tiny covariance — EKF may adjust range based on bearings.
            if len(st.session_state.log) == 0:
                # set initial position but keep covariance normal
                B0 = deg2rad(st.session_state.target_prior["bearing_deg"])
                ekf.x[0] = own["x"] + fv["range"] * math.sin(B0)
                ekf.x[1] = own["y"] + fv["range"] * math.cos(B0)
                # keep P unchanged (so range can evolve)

    # Freeze course: set vx,vy direction to frozen course (use frozen speed if provided)
    if f["course"] and (fv["course"] is not None):
        _,_,_,_,_,sp = ekf.estimate()
        v_use = fv.get("speed", sp)
        v_nm_min = v_use / 60.0
        c_rad = deg2rad(fv["course"])
        ekf.x[2] = v_nm_min * math.sin(c_rad)
        ekf.x[3] = v_nm_min * math.cos(c_rad)
        ekf.P[2,2] = 1e-6; ekf.P[3,3] = 1e-6

    # Freeze speed (magnitude) but allow direction change (so course can still be refined)
    if f["speed"] and (fv["speed"] is not None):
        px, py, vx, vy, crs, sp = ekf.estimate()
        dir_rad = math.atan2(vx, vy)  # current direction
        v_nm_min = fv["speed"]/60.0
        ekf.x[2] = v_nm_min * math.sin(dir_rad)
        ekf.x[3] = v_nm_min * math.cos(dir_rad)
        ekf.P[2,2] = 1e-6; ekf.P[3,3] = 1e-6

def log_row_simple(t_min, bearing_deg):
    ekf = st.session_state.ekf
    own = st.session_state.own
    prior = st.session_state.target_prior

    # ensure first_bearing stored as numeric
    if st.session_state.first_bearing is None:
        st.session_state.first_bearing = float(bearing_deg)
    first_bearing = float(st.session_state.first_bearing)

    R0 = prior["range_nm"]
    B0 = deg2rad(first_bearing)
    dx0 = R0 * math.sin(B0)
    dy0 = R0 * math.cos(B0)
    tgt_v = prior["speed_kn"]/60.0
    if prior["geometry"] == "Opening":
        tgt_cr = B0
    else:
        tgt_cr = (B0 + math.pi) % (2*math.pi)
    px_dr = dx0 + tgt_v * math.sin(tgt_cr) * t_min
    py_dr = dy0 + tgt_v * math.cos(tgt_cr) * t_min

    own_v = own["speed_kn"]/60.0
    ox = own_v * math.sin(deg2rad(own["course_deg"])) * t_min
    oy = own_v * math.cos(deg2rad(own["course_deg"])) * t_min
    dx_dr = px_dr - ox
    dy_dr = py_dr - oy
    range_dr = math.hypot(dx_dr, dy_dr)

    px, py, vx, vy, crs, spd = ekf.estimate()
    dx_ekf = px - own["x"]
    dy_ekf = py - own["y"]
    range_ekf = math.hypot(dx_ekf, dy_ekf)
    pred_brg = tn_bearing_from_dxdy(dx_ekf, dy_ekf)
    err = ((bearing_deg - pred_brg + 180) % 360) - 180

    obs_b_str = f"{int(round(bearing_deg))%360:03d}"
    pred_b_str = f"{int(round(pred_brg))%360:03d}"

    st.session_state.log.append({
        "Time (min)": round(t_min,3),
        "Obs Bearing (°TN)": obs_b_str,
        "Pred Bearing (°TN)": pred_b_str,
        "Obs Bearing (num)": float(bearing_deg),
        "Pred Bearing (num)": float(pred_brg),
        "Bearing Error (°)": round(err,2),
        "DR Range (nm)": round(range_dr,3),
        "EKF Range (nm)": round(range_ekf,3),
        "_px": float(px),
        "_py": float(py),
        "_own_x": float(own["x"]),
        "_own_y": float(own["y"])
    })

# ---------------- radar plot with history ----------------
def plot_radar():
    if not st.session_state.log or st.session_state.ekf is None:
        return

    ekf_rs = [r["EKF Range (nm)"] for r in st.session_state.log]
    ekf_thetas = [deg2rad(r["Pred Bearing (num)"]) for r in st.session_state.log]
    dr_rs = [r["DR Range (nm)"] for r in st.session_state.log]
    dr_thetas = [deg2rad(r["Obs Bearing (num)"]) for r in st.session_state.log]

    ekf = st.session_state.ekf
    px, py, vx, vy, course_est, spd = ekf.estimate()
    dx = px - st.session_state.own["x"]
    dy = py - st.session_state.own["y"]
    cur_r = math.hypot(dx, dy)
    cur_b = deg2rad(tn_bearing_from_dxdy(dx,dy))

    max_r = max(max(ekf_rs, default=1), max(dr_rs, default=1), cur_r) * 1.2
    max_r = max(4, max_r)
    ring_step = max(1, round(max_r / 6.0))
    rings = np.arange(0, max_r + ring_step, ring_step)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rmax(max_r)
    ax.set_rticks(rings)
    ax.set_xticks(np.deg2rad(np.arange(0,360,30)))
    ax.set_xticklabels([f"{d:03d}" for d in range(0,360,30)])

    # plot DR & EKF history
    if dr_rs:
        ax.plot(dr_thetas, dr_rs, 'x-', color='tab:blue', label='DR (obs+DR range)')
    if ekf_rs:
        ax.plot(ekf_thetas, ekf_rs, 'o-', color='tab:red', label='EKF')

    # current EKF
    ax.scatter([cur_b], [cur_r], c='tab:red', s=120, edgecolors='k')

    # ownship center & course arrow (cartesian overlay)
    ax.scatter([0],[0], c='tab:blue', s=80)
    course_rad = deg2rad(st.session_state.own["course_deg"])
    ax_cart = fig.add_axes([0.0,0.0,1,1], polar=False, frameon=False)
    ax_cart.axis('off')
    limit = max_r
    ax_cart.set_xlim(-limit, limit)
    ax_cart.set_ylim(-limit, limit)
    ox = 0; oy = 0
    arrow_dx = 0.8 * math.sin(course_rad)
    arrow_dy = 0.8 * math.cos(course_rad)
    ax_cart.arrow(ox, oy, arrow_dx, arrow_dy, head_width=0.12, head_length=0.12, color='tab:blue')

    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
    st.session_state["plot_placeholder"].pyplot(fig)
    plt.close(fig)

# ---------------- UI ----------------
st.title("⚓ EKF Ship Tracker — fixes: own edits, freeze modes, no reset")

# Sidebar: own ship & priors
with st.sidebar:
    st.header("Own Ship (editing does NOT reset EKF/timer)")
    st.session_state.own["course_deg"] = st.number_input("Own Course (°TN)", 0, 359, int(st.session_state.own["course_deg"]))
    st.session_state.own["speed_kn"] = st.number_input("Own Speed (kn)", 0.0, 60.0, float(st.session_state.own["speed_kn"]), step=0.5)

    st.markdown("---")
    st.header("Target Prior (seed EKF)")
    st.session_state.target_prior["bearing_deg"] = st.number_input("Initial Bearing (°TN)", 0, 359, int(st.session_state.target_prior["bearing_deg"]))
    st.session_state.target_prior["range_nm"] = st.number_input("Initial Range (nm)", 1.0, 200.0, float(st.session_state.target_prior["range_nm"]), step=0.5)
    st.session_state.target_prior["speed_kn"] = st.number_input("Initial Speed (kn)", 0.0, 100.0, float(st.session_state.target_prior["speed_kn"]), step=0.5)
    st.session_state.target_prior["geometry"] = st.radio("Geometry (seed EKF)", ["Opening","Closing"])

    st.markdown("---")
    st.header("Freeze Controls")
    st.session_state.freeze["range"] = st.checkbox("Freeze Range (enable)", value=st.session_state.freeze["range"])
    if st.session_state.freeze["range"]:
        # range freeze mode: Lock vs Anchor
        st.session_state.freeze["range_mode"] = st.radio("Range freeze mode", ["Anchor","Lock"], index=0)
        st.session_state.frozen_vals["range"] = st.number_input("Frozen Range (nm)", 0.1, 2000.0, value=float(st.session_state.target_prior["range_nm"]))
    else:
        st.session_state.frozen_vals["range"] = None
    st.session_state.freeze["speed"] = st.checkbox("Freeze Speed (magnitude)", value=st.session_state.freeze["speed"])
    if st.session_state.freeze["speed"]:
        st.session_state.frozen_vals["speed"] = st.number_input("Frozen Speed (kn)", 0.0, 500.0, value=float(st.session_state.target_prior["speed_kn"]))
    else:
        st.session_state.frozen_vals["speed"] = None
    # course freeze is available after 3 bearings (we enable but warn)
    course_checkbox = st.checkbox("Freeze Course (direction)", value=st.session_state.freeze["course"], disabled=(len(st.session_state.log) < 3))
    st.session_state.freeze["course"] = course_checkbox
    if st.session_state.freeze["course"]:
        st.session_state.frozen_vals["course"] = st.number_input("Frozen Course (°TN)", 0, 359, value=int(st.session_state.frozen_vals.get("course") or 0))
    else:
        st.session_state.frozen_vals["course"] = None

    if st.button("Apply Freeze (no reset)"):
        # apply freeze changes to EKF (Anchor or Lock behavior handled in enforce_freeze_post_update)
        # this call will not reset timer or EKF; it simply enforces current chosen freezes
        st.session_state.freeze["range_mode"] = st.session_state.freeze.get("range_mode", "Anchor")
        enforce_freeze_post_update()
        st.success("Freeze settings applied (EKF state modified accordingly).")

    st.markdown("---")
    if st.button("Initialize Target (seed EKF)"):
        init_ekf_from_priors()
        st.success("EKF seeded from priors. Add bearings to refine solution.")

# Timer display (JS snippet with escaped braces)
start_ms = int(st.session_state.start_time*1000) if (st.session_state.start_time and st.session_state.timer_running) else 0
running_js = "true" if st.session_state.timer_running else "false"
html_timer = f"""
<div style="font-family:monospace;font-size:20px;">Timer: <span id="timer">00:00</span></div>
<script>
const start = {start_ms};
const running = {running_js};
function pad(n) {{return n.toString().padStart(2,'0');}}
if(running && start>0) {{
  function updateTimer() {{
    let now = Date.now();
    let elapsed = Math.max(0, Math.floor((now - start)/1000));
    let m = Math.floor(elapsed/60);
    let s = elapsed%60;
    document.getElementById('timer').innerText = pad(m) + ":" + pad(s);
  }}
  updateTimer();
  setInterval(updateTimer, 1000);
}} else {{
  document.getElementById('timer').innerText = "00:00";
}}
</script>
"""
components.html(html_timer, height=60)

# Timer controls
col1, col2, col3 = st.columns([1,1,2])
with col1:
    if st.button("▶️ Start Timer", disabled=st.session_state.timer_running):
        st.session_state.start_time = time.time()
        st.session_state.timer_running = True
with col2:
    if st.button("⏸ Stop Timer", disabled=not st.session_state.timer_running):
        st.session_state.timer_running = False
with col3:
    st.markdown("Timer is independent — editing Own ship does not restart it.")

# Bearing input (accept 90 or 090)
st.markdown("### Add Bearing (accepts 000 or 90 etc.)")
bearing_str = st.text_input("Bearing (°TN)", value=f"{int(st.session_state.target_prior['bearing_deg']):03d}")
colA, colB = st.columns([1,1])
with colA:
    add_b = st.button("➕ Add Bearing (timestamp now)")
with colB:
    undo = st.button("↶ Undo last bearing")

if undo:
    if st.session_state.log:
        st.session_state.log.pop()
        st.success("Last bearing undone.")
    else:
        st.warning("No bearing to undo.")

if add_b:
    # parse bearing robustly
    s = bearing_str.strip()
    try:
        b = int(s) % 360
    except:
        st.error("Invalid bearing format. Enter 000..359 or 0..359.")
        b = None

    if b is not None:
        if st.session_state.ekf is None:
            st.warning("Initialize EKF first (sidebar).")
        else:
            # compute timestamp (use timer if running)
            if st.session_state.timer_running and st.session_state.start_time:
                elapsed = time.time() - st.session_state.start_time
                tmin = elapsed / 60.0
            else:
                # fallback: one-minute increments
                tmin = st.session_state.last_t_min + 1.0

            dt = max(1e-9, tmin - st.session_state.last_t_min)
            # advance ownship
            own_step(dt)

            # EKF predict & update (using current own pos)
            st.session_state.ekf.predict(dt)
            st.session_state.ekf.update_bearing(deg2rad(b), st.session_state.own["x"], st.session_state.own["y"])

            # enforce freeze AFTER update (Anchor mode for range will not pin)
            enforce_freeze_post_update()

            # log
            log_row_simple(tmin, float(b))
            st.session_state.last_t_min = tmin
            st.success(f"Added bearing {b:03d} at {tmin:.2f} min")

# Outputs (current EKF estimate)
st.markdown("### Results")
if st.session_state.ekf is not None:
    px, py, vx, vy, crs, spd = st.session_state.ekf.estimate()
    rng = math.hypot(px - st.session_state.own["x"], py - st.session_state.own["y"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Target Course (°TN)", f"{crs:.1f}")
    c2.metric("Target Speed (kn)", f"{spd:.2f}")
    c3.metric("Target Range (nm)", f"{rng:.2f}")

# Table (display-friendly)
if st.session_state.log:
    disp = []
    for r in st.session_state.log:
        disp.append({
            "Time (min)": r["Time (min)"],
            "Obs Bearing (°TN)": r["Obs Bearing (°TN)"],
            "Pred Bearing (°TN)": r["Pred Bearing (°TN)"],
            "Bearing Error (°)": r["Bearing Error (°)"],
            "DR Range (nm)": r["DR Range (nm)"],
            "EKF Range (nm)": r["EKF Range (nm)"]
        })
    st.dataframe(disp, use_container_width=True)
    plot_radar()
else:
    st.info("No bearings logged. Seed EKF from sidebar, then add bearings.")
