"""
Orbital Traffic Center — Premium Edition (Option B)
A single-file, runnable Python program.
Features:
 - Tkinter GUI (left control panel, right Matplotlib 3D + 2D views)
 - Textured Earth (optional local image 'earth_texture.jpg')
 - Keplerian orbital propagation (simple 2-body Kepler solver)
 - Continuous collision prediction and minimum-miss-distance calculation
 - Maneuver application with fuel accounting and simple fuel model
 - Lightweight Q-learning (Q-table) agent for auto-avoid decisions
 - Thruster flame visuals, fading trails, mission logging (CSV), export
 - Add / Remove / List satellites dialogs
 - Visual polish: ambient lighting, orbit ribbons, icons, info overlays

Notes before running:
 - Requires: numpy, matplotlib, pillow (PIL), pandas (optional for RL CSV), pyttsx3 (optional for TTS)
 - Place an 'earth_texture.jpg' in the same folder for full texture. If missing, the code draws a shaded Earth.
 - This file intentionally keeps physics simple (Keplerian, no perturbations) so it's easy to run.

Usage: python orbital_traffic_center_premium.py
"""

import os
import math
import time
import csv
import random
import threading
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, filedialog

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Optional dependencies
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# ----------------------------
# Constants & small utils
# ----------------------------
MU = 398600.4418     # km^3 / s^2 (Earth)
R_EARTH = 6378.1363  # km
deg2rad = lambda d: d*math.pi/180.0
rad2deg = lambda r: r*180.0/math.pi

# Safe floating clamps
def clamp(x, a, b):
    return max(a, min(b, x))

# ----------------------------
# Keplerian helper functions
# ----------------------------

def solve_kepler(M, e, tol=1e-9):
    """Solve Kepler's equation using Newton's method for eccentric anomaly E."""
    M = M % (2*math.pi)
    if e < 0.8:
        E = M
    else:
        E = math.pi
    for _ in range(300):
        f = E - e * math.sin(E) - M
        fp = 1 - e * math.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            break
    return E


def true_anomaly_from_E(E, e):
    return 2.0 * math.atan2(math.sqrt(1+e)*math.sin(E/2.0), math.sqrt(1-e)*math.cos(E/2.0))


def perifocal_to_eci(r_pf, inc_deg, raan_deg, argp_deg):
    inc = deg2rad(inc_deg); raan = deg2rad(raan_deg); argp = deg2rad(argp_deg)
    ca = math.cos(argp); sa = math.sin(argp)
    cr = math.cos(raan); sr = math.sin(raan)
    ci = math.cos(inc); si = math.sin(inc)
    R = np.array([
        [cr*ca - sr*sa*ci, -cr*sa - sr*ca*ci, sr*si],
        [sr*ca + cr*sa*ci, -sr*sa + cr*ca*ci, -cr*si],
        [sa*si, ca*si, ci]
    ])
    return R.dot(r_pf)


def orbital_position_from_elements(a_km, ecc, inc_deg, raan_deg, argp_deg, nu_deg):
    nu = deg2rad(nu_deg)
    p = a_km * (1 - ecc**2)
    r_pf = np.array([ (p / (1 + ecc*math.cos(nu))) * math.cos(nu),
                      (p / (1 + ecc*math.cos(nu))) * math.sin(nu),
                      0.0 ])
    return perifocal_to_eci(r_pf, inc_deg, raan_deg, argp_deg)

# ----------------------------
# Satellite class
# ----------------------------
class Satellite:
    def __init__(self, name, a_km, ecc, inc_deg, raan_deg, argp_deg, nu_deg, fuel=1.0, priority=5):
        self.name = name
        self.a = float(a_km)
        self.ecc = float(ecc)
        self.inc = float(inc_deg)
        self.raan = float(raan_deg)
        self.argp = float(argp_deg)
        self.nu = float(nu_deg)
        self.fuel = float(clamp(fuel, 0.0, 1.0))
        self.priority = int(clamp(priority, 1, 10))
        self.trail = []
        self.last_burn = None
        self.color = None

    def position(self):
        return orbital_position_from_elements(self.a, self.ecc, self.inc, self.raan, self.argp, self.nu)

    def mean_motion(self):
        return math.sqrt(MU / (self.a**3))

    def advance_true_anomaly_by_dt(self, dt):
        n = self.mean_motion()
        nu_rad = deg2rad(self.nu)
        E = 2.0 * math.atan2(math.sqrt(1-self.ecc) * math.sin(nu_rad/2.0),
                              math.sqrt(1+self.ecc) * math.cos(nu_rad/2.0))
        M0 = E - self.ecc * math.sin(E)
        M = M0 + n * dt
        E_new = solve_kepler(M, self.ecc)
        nu_new = true_anomaly_from_E(E_new, self.ecc)
        self.nu = (rad2deg(nu_new)) % 360.0

    def copy(self):
        return Satellite(self.name, self.a, self.ecc, self.inc, self.raan, self.argp, self.nu, self.fuel, self.priority)

# ----------------------------
# Maneuvers & fuel model
# ----------------------------

def apply_tangential_delta_v(sat: Satellite, dv_m_s):
    v = math.sqrt(MU / sat.a)
    dv_km_s = dv_m_s / 1000.0
    energy0 = -MU/(2.0*sat.a)
    delta_energy = 0.5 * ((v + dv_km_s)**2 - v**2)
    new_energy = energy0 + delta_energy
    if new_energy >= 0:
        return sat.copy()
    new_a = -MU / (2.0 * new_energy)
    new_sat = sat.copy()
    new_sat.a = new_a
    new_sat.last_burn = (time.time(), dv_m_s, "prograde" if dv_m_s>0 else "retrograde")
    return new_sat


def apply_radial_delta_r(sat: Satellite, delta_r_km):
    new_sat = sat.copy()
    new_sat.argp = (new_sat.argp + (delta_r_km/100.0)) % 360.0
    new_sat.last_burn = (time.time(), 0.0, "radial")
    return new_sat


def apply_normal_delta_inc(sat: Satellite, delta_inc_deg):
    new_sat = sat.copy()
    new_sat.inc = max(0.1, min(179.9, new_sat.inc + delta_inc_deg))
    new_sat.last_burn = (time.time(), 0.0, "normal")
    return new_sat


def fuel_cost_for_dv(dv_m_s):
    return min(1.0, abs(dv_m_s)/50.0)

# ----------------------------
# Prediction & collision detection
# ----------------------------

def predict_positions_for_sat(sat: Satellite, dt_sec, steps):
    tmp = sat.copy()
    positions = [tmp.position()]
    for i in range(1, steps+1):
        tmp.advance_true_anomaly_by_dt(dt_sec)
        positions.append(tmp.position())
    return np.array(positions)


def detect_conjunctions_relvel(preds_list, dt):
    collisions = []
    n = len(preds_list)
    if n < 2:
        return collisions
    steps = preds_list[0].shape[0] - 1
    for i in range(n):
        for j in range(i+1, n):
            best = {"min_dist": float("inf"), "seg_idx": None, "tau": None, "t_from_now": None}
            for k in range(steps):
                r_i_k = preds_list[i][k]; r_i_k1 = preds_list[i][k+1]
                r_j_k = preds_list[j][k]; r_j_k1 = preds_list[j][k+1]
                r0 = r_i_k - r_j_k
                v_rel = (r_i_k1 - r_i_k - (r_j_k1 - r_j_k)) / dt
                vrel2 = np.dot(v_rel, v_rel)
                if vrel2 < 1e-12:
                    d0 = np.linalg.norm(r0)
                    d1 = np.linalg.norm(r_i_k1 - r_j_k1)
                    if d0 < best["min_dist"]:
                        best.update({"min_dist": d0, "seg_idx": k, "tau": 0.0, "t_from_now": k*dt})
                    if d1 < best["min_dist"]:
                        best.update({"min_dist": d1, "seg_idx": k, "tau": dt, "t_from_now": (k+1)*dt})
                else:
                    tau_star = - np.dot(r0, v_rel) / vrel2
                    if 0.0 <= tau_star <= dt:
                        dmin = np.linalg.norm(r0 + v_rel * tau_star)
                        t_from_now = k*dt + tau_star
                        if dmin < best["min_dist"]:
                            best.update({"min_dist": dmin, "seg_idx": k, "tau": tau_star, "t_from_now": t_from_now})
                    d0 = np.linalg.norm(r0)
                    d1 = np.linalg.norm(r_i_k1 - r_j_k1)
                    if d0 < best["min_dist"]:
                        best.update({"min_dist": d0, "seg_idx": k, "tau": 0.0, "t_from_now": k*dt})
                    if d1 < best["min_dist"]:
                        best.update({"min_dist": d1, "seg_idx": k, "tau": dt, "t_from_now": (k+1)*dt})
            collisions.append({
                "pair": (i, j),
                "min_dist_km": float(best["min_dist"]),
                "seg_idx": int(best["seg_idx"]) if best["seg_idx"] is not None else None,
                "tau_in_seg_s": float(best["tau"]) if best["tau"] is not None else None,
                "time_to_CA_s": float(best["t_from_now"]) if best["t_from_now"] is not None else None
            })
    return collisions


def traffic_advice_from_relative_vector_and_ca(rel_vec, min_dist, threshold_km, time_to_ca_s):
    x,y,z = rel_vec; ax, ay, az = abs(x), abs(y), abs(z)
    if az > max(ax, ay):
        direction = "normal" if z > 0 else "anti-normal"
        action = f"{direction} (change inclination)"
    elif ax > ay:
        direction = "radial-out" if x > 0 else "radial-in"
        action = f"{direction} (radial)"
    else:
        direction = "prograde" if y > 0 else "retrograde"
        action = f"{direction} (along-track)"
    if time_to_ca_s is None or time_to_ca_s < 1e-3:
        urgency = 10.0
    else:
        urgency = max(1.0, 30.0 / time_to_ca_s)
    closure = max(0.0, (threshold_km - min_dist))
    scale = 5.0
    dv = min(10.0, max(0.2, (closure / max(1e-6, threshold_km)) * scale * urgency))
    dv = round(float(dv), 3)
    return action, dv

# ----------------------------
# Voice Alerts
# ----------------------------
class VoiceAlert:
    def __init__(self):
        self.enabled = TTS_AVAILABLE
        if TTS_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)
                self.engine.setProperty('volume', 0.9)
            except Exception:
                self.enabled = False
    def speak(self, text):
        if not self.enabled:
            return
        def _s():
            try:
                self.engine.say(text); self.engine.runAndWait()
            except: pass
        threading.Thread(target=_s, daemon=True).start()

# ----------------------------
# Logger
# ----------------------------
class EventLogger:
    def __init__(self, filename="maneuvers_log.csv"):
        self.filename = filename
        self.columns = ["timestamp","event_type","satellite","pair","action","dv_m_s","time_to_CA_s","fuel_before","fuel_after","note"]
        if not os.path.exists(self.filename):
            with open(self.filename, "w", newline="") as f:
                writer = csv.writer(f); writer.writerow(self.columns)
    def log(self, event_type, sat, pair, action, dv_m_s, time_to_CA_s, fuel_before, fuel_after, note=""):
        row = [time.strftime("%Y-%m-%d %H:%M:%S"), event_type, sat, pair, action, dv_m_s, time_to_CA_s, fuel_before, fuel_after, note]
        with open(self.filename, "a", newline="") as f:
            writer = csv.writer(f); writer.writerow(row)

# ----------------------------
# Simple RL agent
# ----------------------------
class SimpleRLEncoder:
    @staticmethod
    def encode(min_dist_km, time_to_ca_s):
        dist_bins = [5, 10, 20, 50, 1e9]
        time_bins = [30, 120, 600, 1e9]
        d_bin = next(i for i,v in enumerate(dist_bins) if min_dist_km <= v)
        t_bin = next(i for i,v in enumerate(time_bins) if (time_to_ca_s or 1e9) <= v)
        return (d_bin, t_bin)

class RlAgent:
    ACTIONS = ["none","prograde","retrograde","radial","normal"]
    def __init__(self, filename="rl_agent_q.csv", alpha=0.3, gamma=0.9, eps=0.2):
        self.filename = filename
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.q = {}
        self._load()
    def _load(self):
        if os.path.exists(self.filename) and PANDAS_AVAILABLE:
            try:
                df = pd.read_csv(self.filename)
                for _,r in df.iterrows():
                    key = (int(r.state_d), int(r.state_t))
                    self.q[key] = [r.q0, r.q1, r.q2, r.q3, r.q4]
            except Exception:
                self.q = {}
    def _save(self):
        rows = []
        for k,v in self.q.items():
            rows.append({"state_d":k[0],"state_t":k[1],"q0":v[0],"q1":v[1],"q2":v[2],"q3":v[3],"q4":v[4]})
        if PANDAS_AVAILABLE:
            pd.DataFrame(rows).to_csv(self.filename, index=False)
        else:
            with open(self.filename, 'w', newline='') as f:
                w = csv.writer(f); w.writerow(["state_d","state_t","q0","q1","q2","q3","q4"])
                for r in rows:
                    w.writerow([r['state_d'], r['state_t'], r['q0'], r['q1'], r['q2'], r['q3'], r['q4']])
    def choose(self, state):
        if random.random() < self.eps or state not in self.q:
            return random.choice(range(len(self.ACTIONS)))
        return int(np.argmax(self.q[state]))
    def update(self, state, action_idx, reward, next_state):
        if state not in self.q:
            self.q[state] = [0.0]*len(self.ACTIONS)
        if next_state not in self.q:
            self.q[next_state] = [0.0]*len(self.ACTIONS)
        q_old = self.q[state][action_idx]
        q_next_max = max(self.q[next_state])
        self.q[state][action_idx] = q_old + self.alpha * (reward + self.gamma * q_next_max - q_old)
        self._save()

# ----------------------------
# Main Application
# ----------------------------
class OrbitalTrafficApp:
    def __init__(self, master):
        self.master = master
        master.title("Orbital Traffic Center — Premium")
        self.voice = VoiceAlert()
        self.logger = EventLogger()
        self.rl_agent = RlAgent()

        # Left control
        self.left = ttk.Frame(master, padding=8); self.left.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(self.left, text="Orbital Traffic Center", font=("Segoe UI",14,"bold")).pack(pady=4)
        ttk.Button(self.left, text="Add Satellite", command=self.add_sat_dialog).pack(fill='x', pady=4)
        ttk.Button(self.left, text="Remove Satellite", command=self.remove_sat_dialog).pack(fill='x', pady=4)
        ttk.Button(self.left, text="List Satellites", command=self.list_satellites).pack(fill='x', pady=4)
        ttk.Separator(self.left).pack(fill='x', pady=6)
        ttk.Label(self.left, text="Prediction parameters").pack(anchor='w')
        ttk.Label(self.left, text="Horizon steps").pack(anchor='w')
        self.pred_steps_var = tk.IntVar(value=120); ttk.Entry(self.left, textvariable=self.pred_steps_var, width=8).pack(anchor='w')
        ttk.Label(self.left, text="Step dt (s)").pack(anchor='w')
        self.pred_dt_var = tk.DoubleVar(value=30.0); ttk.Entry(self.left, textvariable=self.pred_dt_var, width=8).pack(anchor='w')
        ttk.Label(self.left, text="Collision threshold (km)").pack(anchor='w')
        self.threshold_var = tk.DoubleVar(value=15.0); ttk.Entry(self.left, textvariable=self.threshold_var, width=8).pack(anchor='w')
        ttk.Separator(self.left).pack(fill='x', pady=6)
        ttk.Button(self.left, text="Run detection (one step)", command=self.run_detection_once).pack(fill='x', pady=2)
        ttk.Button(self.left, text="Auto-monitor start", command=self.start_auto_monitor).pack(fill='x', pady=2)
        ttk.Button(self.left, text="Auto-monitor stop", command=self.stop_auto_monitor).pack(fill='x', pady=2)
        ttk.Separator(self.left).pack(fill='x', pady=6)
        ttk.Button(self.left, text="Toggle Auto-Avoid (RL)", command=self.toggle_auto_avoid).pack(fill='x', pady=2)
        self.auto_avoid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.left, text="Auto-Avoid Enabled", variable=self.auto_avoid_var).pack(anchor='w', pady=2)
        ttk.Separator(self.left).pack(fill='x', pady=6)
        ttk.Button(self.left, text="Export Maneuvers CSV", command=self.export_csv).pack(fill='x', pady=4)
        ttk.Button(self.left, text="Quit", command=self.quit_app).pack(fill='x', pady=4)

        # info area
        self.info = tk.Text(self.left, width=40, height=16, wrap='word')
        self.info.pack(pady=6)
        voice_status = "ON" if self.voice.enabled else "OFF"
        self.set_info("Ready. Add satellites to start monitoring.\n" + f"Voice alerts: {voice_status}")

        # plot area: 3D + 2D
        self.fig = plt.Figure(figsize=(12,7))
        self.ax3d = self.fig.add_subplot(121, projection='3d')
        self.ax2d = self.fig.add_subplot(122)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # simulation state
        self.satellites = []
        self.time_sim = 0.0
        self.auto_monitoring = False
        self.update_interval_ms = 400
        self.trail_length = 180

        # visuals
        self.color_cycle = plt.cm.tab10(np.linspace(0,1,10))
        self.earth_texture = None
        self._try_load_earth_texture()

        # default sats
        self.add_satellite("SAT-A", a_km=7000, ecc=0.001, inc=51.6, raan=0.0, argp=0.0, nu=0.0, fuel=1.0, priority=7)
        self.add_satellite("SAT-B", a_km=7025, ecc=0.0012, inc=51.6, raan=10.0, argp=0.0, nu=30.0, fuel=0.9, priority=5)
        self.add_satellite("SAT-C", a_km=7050, ecc=0.002, inc=52.0, raan=20.0, argp=0.0, nu=200.0, fuel=0.8, priority=4)

        # drawing initial
        self.draw_once()

    def _try_load_earth_texture(self):
        try:
            tpath = os.path.join(os.path.dirname(__file__), "earth_texture.jpg")
        except NameError:
            tpath = "earth_texture.jpg"
        if os.path.exists(tpath) and PIL_AVAILABLE:
            try:
                img = Image.open(tpath).resize((1024,512))
                self.earth_texture = np.asarray(img)/255.0
            except Exception:
                self.earth_texture = None

    # GUI helpers
    def set_info(self, text):
        self.info.delete('1.0', tk.END); self.info.insert(tk.END, text)

    def add_sat_dialog(self):
        dlg = AddSatelliteDialog(self.master); self.master.wait_window(dlg.top)
        if dlg.result:
            name,a,e,i,raan,argp,nu,fuel,prio = dlg.result
            self.add_satellite(name, a, e, i, raan, argp, nu, fuel, prio)
            self.set_info(f"Added satellite {name}")

    def remove_sat_dialog(self):
        names = [s.name for s in self.satellites]
        if not names:
            messagebox.showinfo("Remove", "No satellites to remove."); return
        sel = simpledialog.askstring("Remove Satellite", f"Enter satellite name to remove:\nAvailable: {', '.join(names)}")
        if sel:
            removed = self.remove_satellite_by_name(sel.strip())
            if removed: self.set_info(f"Removed {sel}")
            else: self.set_info(f"Satellite {sel} not found.")

    def list_satellites(self):
        lines = []
        for s in self.satellites:
            lines.append(f"{s.name}: a={s.a:.1f}km fuel={s.fuel:.2f} prio={s.priority}")
        self.set_info("\n".join(lines))

    def add_satellite(self, name, a_km, ecc, inc, raan, argp, nu, fuel=1.0, priority=5):
        sat = Satellite(name, a_km, ecc, inc, raan, argp, nu, fuel, priority)
        sat.trail = [sat.position()]
        sat.color = tuple(self.color_cycle[len(self.satellites)%10])
        self.satellites.append(sat)
        self.logger.log("create", name, "", "create", 0.0, 0.0, fuel, fuel, "created")

    def remove_satellite_by_name(self, name):
        for idx,s in enumerate(self.satellites):
            if s.name == name:
                del self.satellites[idx]
                self.logger.log("remove", name, "", "remove", 0.0, 0.0, "", "", "removed")
                return True
        return False

    # detection & advice
    def run_detection_once(self):
        if not self.satellites:
            self.set_info("No satellites."); return
        dt = float(self.pred_dt_var.get()); steps = int(self.pred_steps_var.get())
        preds = [predict_positions_for_sat(s, dt, steps) for s in self.satellites]
        pair_results = detect_conjunctions_relvel(preds, dt)
        conjs = [pr for pr in pair_results if pr["min_dist_km"] < float(self.threshold_var.get())]
        if not conjs:
            self.set_info("No conjunctions predicted in horizon."); return

        lines = []
        for c in conjs:
            i,j = c["pair"]; s1 = self.satellites[i]; s2 = self.satellites[j]
            min_d = c["min_dist_km"]; t_to_ca = c.get("time_to_CA_s", None)
            seg = c["seg_idx"]; tau = c["tau_in_seg_s"]
            if seg is not None:
                r_i_k = preds[i][seg]; r_i_k1 = preds[i][seg+1]
                r_j_k = preds[j][seg]; r_j_k1 = preds[j][seg+1]
                vi = (r_i_k1 - r_i_k) / dt
                vj = (r_j_k1 - r_j_k) / dt
                if tau is None: tau = 0.0
                r_i_ca = r_i_k + vi * tau
                r_j_ca = r_j_k + vj * tau
                rel_vec = r_i_ca - r_j_ca
            else:
                rel_vec = preds[i][0] - preds[j][0]
            action_str, dv_sugg = traffic_advice_from_relative_vector_and_ca(rel_vec, min_d, float(self.threshold_var.get()), t_to_ca)
            t_display = int(t_to_ca) if t_to_ca is not None else -1
            lines.append(f"Predicted CA: {s1.name} <-> {s2.name} | miss {min_d:.2f} km | t+{t_display} s | advice: {action_str} | suggested Δv {dv_sugg} m/s")
            if t_to_ca is not None:
                self.voice.speak(f"Collision alert between {s1.name} and {s2.name} in {int(max(1,t_to_ca))} seconds")
            else:
                self.voice.speak(f"Collision alert between {s1.name} and {s2.name}")
            self.logger.log("conjunction", s1.name, s2.name, action_str, dv_sugg, t_to_ca if t_to_ca is not None else -1.0, s1.fuel, s1.fuel, f"min_dist={min_d:.2f}km")
        self.set_info("\n".join(lines))

        if conjs:
            if messagebox.askyesno("Apply maneuvers?", "Apply recommended maneuvers (suggested Δv will be used if fuel allows)?"):
                self.apply_maneuvers_for_conjunctions(conjs, preds)

    def apply_maneuvers_for_conjunctions(self, conjs, preds):
        for c in conjs:
            i,j = c["pair"]; s1 = self.satellites[i]; s2 = self.satellites[j]
            min_d = c["min_dist_km"]; t_to_ca = c.get("time_to_CA_s", None)
            seg = c["seg_idx"]; tau = c["tau_in_seg_s"]
            if seg is not None:
                dt = float(self.pred_dt_var.get())
                r_i_k = preds[i][seg]; r_i_k1 = preds[i][seg+1]
                r_j_k = preds[j][seg]; r_j_k1 = preds[j][seg+1]
                vi = (r_i_k1 - r_i_k) / dt
                vj = (r_j_k1 - r_j_k) / dt
                if tau is None: tau = 0.0
                r_i_ca = r_i_k + vi * tau
                r_j_ca = r_j_k + vj * tau
                rel_vec = r_i_ca - r_j_ca
            else:
                rel_vec = preds[i][0] - preds[j][0]

            action_str, dv_sugg = traffic_advice_from_relative_vector_and_ca(rel_vec, min_d, float(self.threshold_var.get()), t_to_ca)

            auto_action = None
            if self.auto_avoid_var.get():
                state = SimpleRLEncoder.encode(min_d, t_to_ca if t_to_ca is not None else 1e9)
                choice_idx = self.rl_agent.choose(state)
                auto_action = RlAgent.ACTIONS[choice_idx]

            cost_i = (1.0 / max(0.01, s1.fuel)) * (1.0 + (10 - s1.priority)/10.0)
            cost_j = (1.0 / max(0.01, s2.fuel)) * (1.0 + (10 - s2.priority)/10.0)
            mover_idx = i if cost_i < cost_j else j
            mover = self.satellites[mover_idx]
            other = self.satellites[j if mover_idx==i else i]

            chosen_action = auto_action if auto_action and auto_action!="none" else None
            if chosen_action is None:
                if "prograde" in action_str: chosen_action="prograde"
                elif "retrograde" in action_str: chosen_action="retrograde"
                elif "radial" in action_str: chosen_action="radial"
                elif "normal" in action_str: chosen_action="normal"
                else: chosen_action="prograde"

            dv_cmd = 0.0
            new_sat = mover.copy()
            if chosen_action == "prograde":
                dv_cmd = dv_sugg
                new_sat = apply_tangential_delta_v(mover, dv_cmd)
            elif chosen_action == "retrograde":
                dv_cmd = -dv_sugg
                new_sat = apply_tangential_delta_v(mover, dv_cmd)
            elif chosen_action == "radial":
                new_sat = apply_radial_delta_r(mover, +50.0)
            elif chosen_action == "normal":
                new_sat = apply_normal_delta_inc(mover, 0.2)
            else:
                new_sat = apply_tangential_delta_v(mover, dv_sugg); dv_cmd = dv_sugg

            cost_fuel = fuel_cost_for_dv(dv_cmd)
            fuel_before = mover.fuel
            success = False
            if mover.fuel >= cost_fuel and cost_fuel > 0:
                mover.a = new_sat.a; mover.inc = new_sat.inc; mover.argp = new_sat.argp
                mover.fuel = max(0.0, mover.fuel - cost_fuel)
                success = True
                self.logger.log("maneuver", mover.name, other.name, chosen_action, dv_cmd, t_to_ca if t_to_ca is not None else -1.0, fuel_before, mover.fuel, "auto-applied")
                self.set_info(f"Applied {chosen_action} for {mover.name}: dv={dv_cmd} m/s, fuel {fuel_before:.3f}->{mover.fuel:.3f}")
                self.voice.speak(f"Applied maneuver for {mover.name}")
            elif cost_fuel == 0:
                mover.a = new_sat.a; mover.inc = new_sat.inc; mover.argp = new_sat.argp
                self.logger.log("maneuver", mover.name, other.name, chosen_action, dv_cmd, t_to_ca if t_to_ca is not None else -1.0, fuel_before, mover.fuel, "auto-applied (zero dv)")
                self.set_info(f"Applied {chosen_action} for {mover.name} (zero-dv)")
                self.voice.speak(f"Applied maneuver for {mover.name}")
                success = True
            else:
                self.logger.log("maneuver_fail", mover.name, other.name, chosen_action, dv_cmd, t_to_ca if t_to_ca is not None else -1.0, fuel_before, mover.fuel, "insufficient_fuel")
                self.set_info(f"{mover.name} insufficient fuel for dv {dv_cmd} m/s (fuel {mover.fuel:.3f})")
                self.voice.speak(f"{mover.name} insufficient fuel for maneuver")

            if self.auto_avoid_var.get():
                if success:
                    reward = 1.0
                else:
                    reward = -0.5
                prev_state = SimpleRLEncoder.encode(min_d, t_to_ca if t_to_ca is not None else 1e9)
                next_state = SimpleRLEncoder.encode(min(1e9, min_d + (10.0 if success else -2.0)), (t_to_ca or 1e9))
                action_idx = RlAgent.ACTIONS.index(chosen_action) if chosen_action in RlAgent.ACTIONS else 0
                self.rl_agent.update(prev_state, action_idx, reward, next_state)

        self.draw_once()

    # auto monitoring
    def start_auto_monitor(self):
        if self.auto_monitoring: return
        self.auto_monitoring = True; self.voice.speak("Auto monitoring started."); self._auto_monitor_loop()

    def stop_auto_monitor(self):
        self.auto_monitoring = False; self.voice.speak("Auto monitoring stopped.")

    def _auto_monitor_loop(self):
        if not self.auto_monitoring: return
        self.run_detection_once()
        dt_adv = float(self.pred_dt_var.get())
        for s in self.satellites:
            s.advance_true_anomaly_by_dt(dt_adv)
            pos = s.position(); s.trail.append(pos)
            if len(s.trail) > self.trail_length: s.trail.pop(0)
        self.draw_once()
        self.master.after(self.update_interval_ms, self._auto_monitor_loop)

    # plotting
    def draw_once(self):
        self.ax3d_clear_and_draw(); self.ax2d_clear_and_draw(); self.canvas.draw()

    def ax3d_clear_and_draw(self):
        self.ax3d.clear()
        u = np.linspace(0, 2*np.pi, 120); v = np.linspace(0, np.pi, 60)
        xu = (R_EARTH) * np.outer(np.cos(u), np.sin(v))
        yu = (R_EARTH) * np.outer(np.sin(u), np.sin(v))
        zu = (R_EARTH) * np.outer(np.ones_like(u), np.cos(v))
        sun = np.array([1.0, 1.0, 0.5]); sun = sun / np.linalg.norm(sun)
        normals = np.stack([xu, yu, zu], axis=-1)
        normals_unit = normals / np.linalg.norm(normals, axis=-1)[...,None]
        lambert = np.clip(np.tensordot(normals_unit, sun, axes=([2],[0])), 0.0, 1.0)
        if self.earth_texture is not None:
            base_color = np.zeros(lambert.shape + (3,))
            base_color[...,0] = 0.2 + 0.8*lambert
            base_color[...,1] = 0.45 + 0.55*lambert
            base_color[...,2] = 0.9*lambert
            self.ax3d.plot_surface(xu, yu, zu, facecolors=base_color, linewidth=0, antialiased=False, shade=False)
        else:
            self.ax3d.plot_surface(xu, yu, zu, facecolors=plt.cm.terrain(lambert), linewidth=0, antialiased=False, shade=False, alpha=0.95)

        for idx, s in enumerate(self.satellites):
            pts = predict_positions_for_sat(s, dt_sec=60.0, steps=180)
            self.ax3d.plot(pts[:,0], pts[:,1], pts[:,2], linewidth=1.1, alpha=0.6, color=s.color)
            cur = s.position()
            self.ax3d.scatter([cur[0]], [cur[1]], [cur[2]], s=80, color=s.color, edgecolors='k')
            if len(s.trail) > 1:
                tr = np.array(s.trail)
                L = tr.shape[0]
                for k in range(1, L):
                    a = (k/L)**1.2
                    self.ax3d.plot([tr[k-1,0], tr[k,0]], [tr[k-1,1], tr[k,1]], [tr[k-1,2], tr[k,2]], color=s.color, alpha=a, linewidth=2)
            if s.last_burn and (time.time() - s.last_burn[0]) < 3.0:
                burn_time, dv, action = s.last_burn
                pos = np.array(cur)
                dir_vec = pos / (np.linalg.norm(pos) + 1e-9)
                flame_tip = pos - dir_vec * (R_EARTH * 0.05)
                flame_base = pos - dir_vec * (R_EARTH * 0.015)
                xs = [flame_base[0], flame_tip[0], flame_base[0]]
                ys = [flame_base[1], flame_tip[1], flame_base[1]]
                zs = [flame_base[2], flame_tip[2], flame_base[2]]
                self.ax3d.plot(xs, ys, zs, color='orange', linewidth=3, alpha=0.9)

        self.ax3d.set_box_aspect([1,1,1])
        self.ax3d.set_xlabel("X km"); self.ax3d.set_ylabel("Y km"); self.ax3d.set_zlabel("Z km")
        self.ax3d.view_init(elev=25, azim=(time.time() % 360)*5 % 360)

    def ax2d_clear_and_draw(self):
        self.ax2d.clear()
        colors = plt.cm.tab10(np.linspace(0,1,max(1,len(self.satellites))))
        circle = plt.Circle((0,0), R_EARTH, color=(0.12,0.2,0.5), alpha=0.8)
        self.ax2d.add_artist(circle)
        max_range = R_EARTH + 10000
        for idx, s in enumerate(self.satellites):
            pts = predict_positions_for_sat(s, dt_sec=self.pred_dt_var.get(), steps=120)
            xs = pts[:,0]; ys = pts[:,1]
            self.ax2d.plot(xs, ys, color=s.color, linewidth=1.2, alpha=0.9)
            cur = s.position()
            self.ax2d.scatter(cur[0], cur[1], marker='o', s=50, color=s.color, edgecolors='k')
            self.ax2d.text(cur[0], cur[1], f" {s.name} f={s.fuel:.2f}", fontsize=8, color='w', bbox=dict(facecolor='black', alpha=0.4, pad=1))
        self.ax2d.set_aspect('equal', adjustable='datalim')
        self.ax2d.set_xlim(-max_range, max_range); self.ax2d.set_ylim(-max_range, max_range)
        self.ax2d.set_xlabel("X km"); self.ax2d.set_ylabel("Y km")
        self.ax2d.set_title("Top-down (XY) prediction")

    # export / quit
    def export_csv(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if not path: return
        if not os.path.exists(self.logger.filename):
            messagebox.showinfo("Export", "No maneuvers logged yet."); return
        try:
            import shutil; shutil.copy(self.logger.filename, path)
            messagebox.showinfo("Export", f"Saved maneuvers log to {path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def quit_app(self):
        if messagebox.askokcancel("Quit", "Exit the Orbital Traffic Center?"):
            self.master.destroy()

    def toggle_auto_avoid(self):
        self.auto_avoid_var.set(not self.auto_avoid_var.get())
        self.set_info(f"Auto-avoid set to {self.auto_avoid_var.get()}")

# Dialog for adding satellites
class AddSatelliteDialog:
    def __init__(self, parent):
        top = self.top = tk.Toplevel(parent); top.title("Add Satellite")
        top.grab_set()
        ttk.Label(top, text="Name").grid(row=0, column=0)
        self.name = tk.StringVar(value=f"SAT-{random.randint(100,999)}"); ttk.Entry(top, textvariable=self.name).grid(row=0, column=1)
        ttk.Label(top, text="a (km)").grid(row=1,column=0); self.a = tk.DoubleVar(value=7000.0); ttk.Entry(top, textvariable=self.a).grid(row=1,column=1)
        ttk.Label(top, text="ecc").grid(row=2,column=0); self.e = tk.DoubleVar(value=0.001); ttk.Entry(top, textvariable=self.e).grid(row=2,column=1)
        ttk.Label(top, text="inc (deg)").grid(row=3,column=0); self.inc = tk.DoubleVar(value=51.6); ttk.Entry(top, textvariable=self.inc).grid(row=3,column=1)
        ttk.Label(top, text="raan (deg)").grid(row=4,column=0); self.raan = tk.DoubleVar(value=0.0); ttk.Entry(top, textvariable=self.raan).grid(row=4,column=1)
        ttk.Label(top, text="argp (deg)").grid(row=5,column=0); self.argp = tk.DoubleVar(value=0.0); ttk.Entry(top, textvariable=self.argp).grid(row=5,column=1)
        ttk.Label(top, text="nu (deg)").grid(row=6,column=0); self.nu = tk.DoubleVar(value=0.0); ttk.Entry(top, textvariable=self.nu).grid(row=6,column=1)
        ttk.Label(top, text="fuel (0..1)").grid(row=7,column=0); self.fuel = tk.DoubleVar(value=1.0); ttk.Entry(top, textvariable=self.fuel).grid(row=7,column=1)
        ttk.Label(top, text="priority (1..10)").grid(row=8,column=0); self.prio = tk.IntVar(value=5); ttk.Entry(top, textvariable=self.prio).grid(row=8,column=1)
        ttk.Button(top, text="Add", command=self.on_add).grid(row=9,column=0, pady=6); ttk.Button(top, text="Cancel", command=self.on_cancel).grid(row=9,column=1)
        self.result = None
    def on_add(self):
        try:
            self.result = (self.name.get(), float(self.a.get()), float(self.e.get()), float(self.inc.get()), float(self.raan.get()), float(self.argp.get()), float(self.nu.get()), float(self.fuel.get()), int(self.prio.get()))
        except Exception as e:
            messagebox.showerror("Invalid Input", str(e)); return
        self.top.destroy()
    def on_cancel(self): self.top.destroy()

# ----------------------------
# Run
# ----------------------------
def main():
    root = tk.Tk()
    app = OrbitalTrafficApp(root)
    root.geometry("1400x820")
    root.mainloop()

if __name__ == "__main__":
    main()
