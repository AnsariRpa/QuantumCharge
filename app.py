"""
QuantumCharge 2.5 — Dynamic route-aware EV charging optimization.

Journey labels and station records are user-entered or synthetic. Route position and travel time are simplified estimates.
No live maps, traffic, charging networks or quantum hardware are queried.
"""

from __future__ import annotations

import base64
import itertools
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

st.set_page_config(
    page_title="QuantumCharge | Dynamic EV Optimization",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
HERO = ROOT / "assets" / "quantumcharge_hero.png"
MAX_QUBITS = 22
MAX_CLASSICAL_COMBINATIONS = 250_000

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
:root{--qc-bg:#050b18;--qc-panel:#0a1730;--qc-blue:#168cff;--qc-cyan:#00e5ff;--qc-text:#edf6ff;}
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:radial-gradient(ellipse at 12% 0%,rgba(0,116,255,.16),transparent 34%),radial-gradient(ellipse at 92% 14%,rgba(0,229,255,.10),transparent 28%),linear-gradient(145deg,#040916 0%,#071329 48%,#040a17 100%);color:var(--qc-text);}
[data-testid="stHeader"]{background:rgba(4,9,22,.75);}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#06132a 0%,#071a36 55%,#040d1d 100%);border-right:1px solid rgba(0,190,255,.28);}
[data-testid="stSidebar"] *{color:#e8f5ff;}
.block-container{padding-top:1.6rem;padding-bottom:3rem;max-width:1500px;}
h1,h2,h3{color:#f4f9ff!important;letter-spacing:-.035em;}
p,li,label,.stMarkdown{color:#d4e5fa;}
div[data-testid="stMetric"]{background:linear-gradient(145deg,rgba(13,34,67,.95),rgba(7,20,42,.95));border:1px solid rgba(0,190,255,.25);border-radius:16px;padding:16px 18px;box-shadow:0 0 22px rgba(0,120,255,.08);}
div[data-testid="stMetric"] label{color:#8fb9df!important;}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{color:#f2fbff!important;}
.stButton>button{border:1px solid #00dfff;border-radius:12px;color:#00101e;background:linear-gradient(100deg,#00dfff,#168cff);font-weight:800;min-height:2.8rem;box-shadow:0 0 20px rgba(0,205,255,.25);}
.stButton>button:hover{border-color:#8af4ff;box-shadow:0 0 30px rgba(0,205,255,.5);color:#00101e;}
div[data-testid="stExpander"]{border:1px solid rgba(0,190,255,.24);border-radius:14px;background:rgba(8,24,48,.70);}
div[data-testid="stDataFrame"],div[data-testid="stDataEditor"]{border:1px solid rgba(0,190,255,.20);border-radius:12px;overflow:hidden;}
hr{border-color:rgba(0,190,255,.20);}
.qc-eyebrow{color:#00e5ff;text-transform:uppercase;font-size:.76rem;font-weight:800;letter-spacing:.18em;margin-bottom:.55rem;}
.qc-hero{position:relative;overflow:hidden;min-height:370px;border:1px solid rgba(0,204,255,.42);border-radius:24px;padding:42px 44px;background-size:cover;background-position:center right;box-shadow:0 0 38px rgba(0,120,255,.18);display:flex;align-items:center;}
.qc-hero:before{content:"";position:absolute;inset:0;background:linear-gradient(90deg,rgba(3,10,27,.98) 0%,rgba(3,13,34,.9) 42%,rgba(3,13,34,.15) 100%);}
.qc-hero-content{position:relative;z-index:1;max-width:660px;}
.qc-title{font-size:clamp(2.5rem,5vw,4.5rem);line-height:1.02;font-weight:900;color:#f5fbff;letter-spacing:-.065em;margin:0;}
.qc-title span{background:linear-gradient(90deg,#fff,#55eaff 55%,#168cff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.qc-subtitle{color:#d7edff;font-size:1.1rem;line-height:1.65;max-width:590px;margin-top:18px;}
.qc-chip-row{display:flex;flex-wrap:wrap;gap:10px;margin-top:25px;}
.qc-chip{color:#c8f8ff;border:1px solid rgba(0,220,255,.45);background:rgba(0,118,190,.17);border-radius:999px;padding:8px 13px;font-size:.78rem;font-weight:700;}
.qc-card{height:100%;min-height:145px;border:1px solid rgba(0,190,255,.28);border-radius:18px;padding:20px;background:linear-gradient(145deg,rgba(13,37,72,.94),rgba(7,19,40,.94));}
.qc-card-icon{font-size:1.7rem;margin-bottom:9px;}
.qc-card-title{color:#f2fbff;font-size:1.02rem;font-weight:800;margin-bottom:7px;}
.qc-card-text{color:#a9c2df;font-size:.88rem;line-height:1.55;}
.qc-section-title{color:#f3f9ff;font-size:1.4rem;font-weight:800;margin:1.5rem 0 .9rem;}
.qc-footer{color:#7794b8;font-size:.76rem;text-align:center;padding:22px 0 4px;}
.qc-note{border:1px solid rgba(0,210,255,.28);background:rgba(0,100,170,.12);border-radius:13px;padding:13px 15px;color:#b9dcf8;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def hero_data() -> str:
    if not HERO.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(HERO.read_bytes()).decode("utf-8")


def page_heading(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="qc-eyebrow">{eyebrow}</div><h1 style="margin-bottom:.25rem">{title}</h1>'
        f'<p style="color:#9db8d8;margin-top:0">{description}</p>',
        unsafe_allow_html=True,
    )


def default_evs(count: int = 2) -> pd.DataFrame:
    seed = [
        ["EV-01", 60.0, 22.0, 70.0, 3],
        ["EV-02", 75.0, 34.0, 80.0, 1],
        ["EV-03", 65.0, 45.0, 80.0, 0],
        ["EV-04", 80.0, 30.0, 75.0, 0],
    ]
    rows = [seed[i] if i < len(seed) else [f"EV-{i+1:02d}", 65.0, 30.0, 80.0, 0] for i in range(count)]
    return pd.DataFrame(rows, columns=["EV ID", "Battery capacity (kWh)", "Current SOC (%)", "Target SOC (%)", "Recent DC fast-charge streak"])


def default_stations(slot_count: int = 2, count: int = 3) -> pd.DataFrame:
    seed = [
        ["ST-A", "Highway Hub A", "DC Fast", 60.0, 5, 2, 1, 75.0],
        ["ST-B", "Travel Plaza B", "DC Fast", 90.0, 10, 5, 1, 155.0],
        ["ST-C", "Community Charger C", "AC", 22.0, 15, 3, 1, 235.0],
        ["ST-D", "Express Charge D", "DC Fast", 120.0, 8, 4, 2, 190.0],
    ]
    rows = []
    for i in range(count):
        row = list(seed[i] if i < len(seed) else [f"ST-{i+1:02d}", f"Synthetic Station {i+1}", "DC Fast", 60.0, 10, 3, 1, 100.0 + i * 40])
        for slot in range(slot_count):
            queue = [18, 8, 6][slot % 3] if i == 0 else [5, 12, 7][slot % 3]
            tariff = [18.0, 20.0, 17.0][slot % 3] if i == 0 else [19.0, 16.0, 18.0][slot % 3]
            row.extend([queue, tariff])
        rows.append(row)
    columns = ["Station ID", "Station name", "Charger type", "Power (kW)", "Detour (min)", "Amenities (1-5)", "Capacity", "Route position (km from origin)"]
    for slot in range(slot_count):
        columns.extend([f"Queue slot {slot+1} (min)", f"Tariff slot {slot+1} (₹/kWh)"])
    return pd.DataFrame(rows, columns=columns)


def make_slots(start_time, count: int, duration: int) -> list[str]:
    start = datetime.combine(datetime.today().date(), start_time)
    return [
        f"{(start + timedelta(minutes=i*duration)).strftime('%H:%M')}–"
        f"{(start + timedelta(minutes=(i+1)*duration)).strftime('%H:%M')}"
        for i in range(count)
    ]


def validate_inputs(ev_df: pd.DataFrame, station_df: pd.DataFrame, slot_count: int, route_distance_km: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ev = ev_df.copy().reset_index(drop=True)
    stations = station_df.copy().reset_index(drop=True)
    if ev.empty or stations.empty:
        raise ValueError("At least one EV and one charging station are required.")
    ev["EV ID"] = [str(v).strip() if pd.notna(v) and str(v).strip() else f"EV-{i+1:02d}" for i, v in enumerate(ev["EV ID"])]
    stations["Station ID"] = [str(v).strip() if pd.notna(v) and str(v).strip() else f"ST-{i+1:02d}" for i, v in enumerate(stations["Station ID"])]
    stations["Station name"] = [str(v).strip() if pd.notna(v) and str(v).strip() else f"Synthetic Station {i+1}" for i, v in enumerate(stations["Station name"])]
    ev_numeric = ["Battery capacity (kWh)", "Current SOC (%)", "Target SOC (%)", "Recent DC fast-charge streak"]
    station_numeric = ["Power (kW)", "Detour (min)", "Amenities (1-5)", "Capacity", "Route position (km from origin)"]
    for slot in range(slot_count):
        station_numeric.extend([f"Queue slot {slot+1} (min)", f"Tariff slot {slot+1} (₹/kWh)"])
    for col in ev_numeric:
        ev[col] = pd.to_numeric(ev[col], errors="coerce")
    for col in station_numeric:
        stations[col] = pd.to_numeric(stations[col], errors="coerce")
    if ev[ev_numeric].isna().any().any() or stations[station_numeric].isna().any().any():
        raise ValueError("Enter valid numeric values in all EV and station fields.")
    if ev["EV ID"].duplicated().any() or stations["Station ID"].duplicated().any():
        raise ValueError("EV IDs and Station IDs must be unique.")
    if (ev["Battery capacity (kWh)"] <= 0).any():
        raise ValueError("Battery capacity must be greater than zero.")
    if ((ev["Current SOC (%)"] < 0) | (ev["Current SOC (%)"] > 100)).any():
        raise ValueError("Current SOC must be between 0 and 100.")
    if ((ev["Target SOC (%)"] <= ev["Current SOC (%)"]) | (ev["Target SOC (%)"] > 100)).any():
        raise ValueError("Target SOC must be higher than current SOC and no more than 100%.")
    if (ev["Recent DC fast-charge streak"] < 0).any():
        raise ValueError("Fast-charge streak cannot be negative.")
    if (stations["Power (kW)"] <= 0).any() or (stations["Capacity"] < 1).any():
        raise ValueError("Station power must be positive and capacity must be at least one.")
    if ((stations["Amenities (1-5)"] < 1) | (stations["Amenities (1-5)"] > 5)).any():
        raise ValueError("Amenities must be rated from 1 to 5.")
    if (stations[station_numeric] < 0).any().any():
        raise ValueError("Queue, tariff, detour, capacity and route position values cannot be negative.")
    if route_distance_km <= 0:
        raise ValueError("Route distance must be greater than zero.")
    if (stations["Route position (km from origin)"] > route_distance_km).any():
        raise ValueError("A station route position cannot exceed the configured journey distance.")
    stations["Capacity"] = stations["Capacity"].astype(int)
    ev["Recent DC fast-charge streak"] = ev["Recent DC fast-charge streak"].astype(int)
    return ev, stations


def build_options(ev_df: pd.DataFrame, station_df: pd.DataFrame, slots: list[str], battery_penalty: float, route_distance_km: float, average_speed_kmph: float) -> pd.DataFrame:
    rows = []
    for _, ev in ev_df.iterrows():
        energy = float(ev["Battery capacity (kWh)"]) * (float(ev["Target SOC (%)"]) - float(ev["Current SOC (%)"])) / 100.0
        for _, station in station_df.iterrows():
            for slot_idx, slot_label in enumerate(slots):
                power = float(station["Power (kW)"])
                # Simplified effective power and fixed connection overhead; not an OEM charge curve.
                charge_min = (energy / (power * 0.82)) * 60.0 + 5.0
                queue = float(station[f"Queue slot {slot_idx+1} (min)"])
                tariff = float(station[f"Tariff slot {slot_idx+1} (₹/kWh)"])
                care_penalty = battery_penalty if str(station["Charger type"]).lower().startswith("dc") and int(ev["Recent DC fast-charge streak"]) >= 4 else 0.0
                route_position = float(station["Route position (km from origin)"])
                approach_min = route_position / max(average_speed_kmph, 1.0) * 60.0
                remaining_km = max(route_distance_km - route_position, 0.0)
                cost = energy * tariff
                amenity_penalty = (5.0 - float(station["Amenities (1-5)"])) * 3.0
                score = approach_min*0.35 + float(station["Detour (min)"]) + charge_min + queue*1.25 + cost*0.025 + amenity_penalty + care_penalty
                rows.append({
                    "ev_id": str(ev["EV ID"]), "station_id": str(station["Station ID"]),
                    "station": str(station["Station name"]), "charger": str(station["Charger type"]),
                    "slot_idx": slot_idx, "slot": slot_label, "capacity": int(station["Capacity"]),
                    "power_kw": power, "route_position_km": route_position, "approach_min": approach_min,
                    "remaining_km": remaining_km, "detour_min": float(station["Detour (min)"]),
                    "charge_min": charge_min, "wait_min": queue, "energy_kwh": energy,
                    "energy_cost": cost, "amenity_score": int(station["Amenities (1-5)"]),
                    "battery_penalty": care_penalty, "score": score,
                })
    return pd.DataFrame(rows)


@dataclass
class Qubo:
    linear: np.ndarray
    quadratic: dict[tuple[int, int], float]
    constant: float
    n: int
    decision_count: int
    slack_map: dict[tuple[str, int], list[tuple[int, int]]]

    def energy(self, bits: np.ndarray) -> float:
        value = self.constant + float(np.dot(self.linear, bits))
        for (i, j), coefficient in self.quadratic.items():
            value += coefficient * bits[i] * bits[j]
        return float(value)

    def all_energies(self) -> np.ndarray:
        if self.n > MAX_QUBITS:
            raise ValueError(
                f"Local Statevector mode is configured for at most {MAX_QUBITS} qubits; "
                f"this QUBO requires {self.n}."
            )

        # QUBO energy is diagonal in the computational basis. Calculate all
        # basis-state energies with NumPy vector operations instead of a
        # Python loop over every one of the 2**n bitstrings.
        state_count = 1 << self.n
        states = np.arange(state_count, dtype=np.uint32)
        energies = np.full(state_count, self.constant, dtype=np.float64)

        for i, coefficient in enumerate(self.linear):
            if abs(float(coefficient)) > 1e-12:
                bit_i = (states >> np.uint32(i)) & np.uint32(1)
                energies += float(coefficient) * bit_i

        for (i, j), coefficient in self.quadratic.items():
            if abs(float(coefficient)) > 1e-12:
                bit_i = (states >> np.uint32(i)) & np.uint32(1)
                bit_j = (states >> np.uint32(j)) & np.uint32(1)
                energies += float(coefficient) * bit_i * bit_j

        return energies


def build_qubo(options: pd.DataFrame, stations: pd.DataFrame) -> Qubo:
    decision_count = len(options)
    slack_map: dict[tuple[str, int], list[tuple[int, int]]] = {}
    next_index = decision_count
    for _, station in stations.iterrows():
        sid = str(station["Station ID"])
        capacity = int(station["Capacity"])
        bit_count = max(1, math.ceil(math.log2(capacity + 1)))
        for slot_idx in sorted(options["slot_idx"].unique()):
            entries = []
            for power in range(bit_count):
                entries.append((next_index, 2 ** power))
                next_index += 1
            slack_map[(sid, int(slot_idx))] = entries

    n = next_index
    linear = np.zeros(n, dtype=float)
    linear[:decision_count] = options["score"].to_numpy(dtype=float)
    quadratic: dict[tuple[int, int], float] = {}
    constant = 0.0
    penalty = max(200.0, float(options["score"].max()) * (len(options["ev_id"].unique()) + 2) * 2.0)

    def add_square(terms: list[tuple[int, float]], target: float) -> None:
        nonlocal constant
        constant += penalty * target * target
        for i, ci in terms:
            linear[i] += penalty * (ci*ci - 2.0*target*ci)
        for pos, (i, ci) in enumerate(terms):
            for j, cj in terms[pos+1:]:
                key = (min(i, j), max(i, j))
                quadratic[key] = quadratic.get(key, 0.0) + 2.0*penalty*ci*cj

    # Exactly one station/time option per EV.
    for ev_id in options["ev_id"].unique():
        indices = options.index[options["ev_id"] == ev_id].tolist()
        add_square([(int(i), 1.0) for i in indices], 1.0)

    # Station/time capacity: sum(assignments) + binary slack = capacity.
    capacity_lookup = stations.set_index("Station ID")["Capacity"].to_dict()
    for (sid, slot_idx), slack_bits in slack_map.items():
        indices = options.index[(options["station_id"] == sid) & (options["slot_idx"] == slot_idx)].tolist()
        terms = [(int(i), 1.0) for i in indices] + slack_bits
        add_square(terms, float(capacity_lookup[sid]))

    return Qubo(linear, quadratic, constant, n, decision_count, slack_map)


def assignment_feasible(bits: np.ndarray, options: pd.DataFrame, stations: pd.DataFrame) -> bool:
    selected = np.flatnonzero(bits[:len(options)])
    for ev_id in options["ev_id"].unique():
        if sum(options.loc[i, "ev_id"] == ev_id for i in selected) != 1:
            return False
    counts = options.iloc[selected].groupby(["station_id", "slot_idx"]).size().to_dict()
    capacities = stations.set_index("Station ID")["Capacity"].to_dict()
    return all(count <= int(capacities[sid]) for (sid, _), count in counts.items())


def complete_bits(selected_indices: list[int], qubo: Qubo, options: pd.DataFrame, stations: pd.DataFrame) -> np.ndarray:
    bits = np.zeros(qubo.n, dtype=int)
    bits[selected_indices] = 1
    occupancy = options.iloc[selected_indices].groupby(["station_id", "slot_idx"]).size().to_dict()
    capacities = stations.set_index("Station ID")["Capacity"].to_dict()
    for (sid, slot_idx), slack_bits in qubo.slack_map.items():
        remaining = int(capacities[sid]) - int(occupancy.get((sid, slot_idx), 0))
        for bit_index, weight in slack_bits:
            if remaining & weight:
                bits[bit_index] = 1
    return bits


def exact_classical(qubo: Qubo, options: pd.DataFrame, stations: pd.DataFrame) -> tuple[np.ndarray, float]:
    groups = [options.index[options["ev_id"] == ev_id].tolist() for ev_id in options["ev_id"].unique()]
    combinations = math.prod(len(group) for group in groups)
    if combinations > MAX_CLASSICAL_COMBINATIONS:
        raise ValueError(f"Exact classical benchmark requires {combinations:,} combinations; limit is {MAX_CLASSICAL_COMBINATIONS:,}.")
    capacities = stations.set_index("Station ID")["Capacity"].to_dict()
    best_bits, best_score = None, float("inf")
    for choice in itertools.product(*groups):
        selected = options.loc[list(choice)]
        occupancy = selected.groupby(["station_id", "slot_idx"]).size().to_dict()
        if any(count > int(capacities[sid]) for (sid, _), count in occupancy.items()):
            continue
        score = float(selected["score"].sum())
        if score < best_score:
            best_score = score
            best_bits = complete_bits(list(choice), qubo, options, stations)
    if best_bits is None:
        raise RuntimeError("No feasible classical assignment exists for this scenario.")
    return best_bits, qubo.energy(best_bits)


def qaoa_circuit(qubo: Qubo, gamma: float, beta: float, measure: bool = False) -> QuantumCircuit:
    circuit = QuantumCircuit(qubo.n, qubo.n if measure else 0)
    circuit.h(range(qubo.n))
    for q, coefficient in enumerate(qubo.linear):
        if abs(coefficient) > 1e-12:
            circuit.p(-gamma * float(coefficient), q)
    for (i, j), coefficient in qubo.quadratic.items():
        if abs(coefficient) > 1e-12:
            circuit.cp(-gamma * float(coefficient), i, j)
    for q in range(qubo.n):
        circuit.rx(2.0 * beta, q)
    if measure:
        circuit.measure(range(qubo.n), range(qubo.n))
    return circuit


def run_qaoa(qubo: Qubo, options: pd.DataFrame, stations: pd.DataFrame, iterations: int, shots: int) -> dict:
    all_energies = qubo.all_energies()

    def expectation(params: np.ndarray) -> float:
        gamma, beta = params
        state = Statevector.from_instruction(qaoa_circuit(qubo, float(gamma), float(beta)))
        return float(np.dot(state.probabilities(), all_energies))

    optimized = minimize(
        expectation, x0=np.array([0.035, 0.55]), method="COBYLA",
        options={"maxiter": iterations, "rhobeg": 0.15, "tol": 1e-3},
    )
    gamma, beta = optimized.x
    circuit = qaoa_circuit(qubo, float(gamma), float(beta), measure=True)
    simulator = AerSimulator()
    counts = simulator.run(transpile(circuit, simulator), shots=shots).result().get_counts()
    best_bits, best_energy, feasible_shots = None, float("inf"), 0
    for bitstring, frequency in counts.items():
        bits = np.array([int(bit) for bit in bitstring.replace(" ", "")[::-1]], dtype=int)
        if assignment_feasible(bits, options, stations):
            selected_score = float(options.iloc[np.flatnonzero(bits[:len(options)])]["score"].sum())
            energy = qubo.energy(bits)
            # Slack bits must satisfy the capacity equality, not merely the assignment inequality.
            if abs(energy - selected_score) < 1e-5:
                feasible_shots += frequency
                if energy < best_energy:
                    best_bits, best_energy = bits, energy
    fallback = False
    if best_bits is None:
        best_bits, best_energy = exact_classical(qubo, options, stations)
        fallback = True
    return {
        "bits": best_bits, "energy": best_energy, "counts": counts,
        "expectation": float(optimized.fun), "gamma": float(gamma), "beta": float(beta),
        "valid_sample_fraction": feasible_shots / shots, "used_fallback": fallback,
        "optimizer_success": bool(optimized.success), "optimizer_message": str(optimized.message),
        "shots": shots,
    }


def run_scenario(ev_df: pd.DataFrame, station_df: pd.DataFrame, slots: list[str], iterations: int, shots: int, battery_penalty: float, route_distance_km: float, average_speed_kmph: float) -> dict:
    options = build_options(ev_df, station_df, slots, battery_penalty, route_distance_km, average_speed_kmph)
    qubo = build_qubo(options, station_df)
    if qubo.n > MAX_QUBITS:
        raise ValueError(
            f"This scenario creates {qubo.n} total QUBO qubits, including capacity slack bits. "
            f"The current local Statevector limit is {MAX_QUBITS}. Reduce EVs, stations, slots or station capacity."
        )
    classical_bits, classical_energy = exact_classical(qubo, options, station_df)
    qaoa = run_qaoa(qubo, options, station_df, iterations, shots)
    return {
        "options": options, "qubo": qubo, "classical_bits": classical_bits,
        "classical_energy": classical_energy, "qaoa": qaoa,
        "classical_selected": options.iloc[np.flatnonzero(classical_bits[:len(options)])].copy(),
        "qaoa_selected": options.iloc[np.flatnonzero(qaoa["bits"][:len(options)])].copy(),
        "evs": ev_df.copy(), "stations": station_df.copy(), "slots": slots,
    }


# Persist editable inputs across Streamlit reruns.
if "qc_evs" not in st.session_state:
    st.session_state.qc_evs = default_evs()
if "qc_stations" not in st.session_state:
    st.session_state.qc_stations = default_stations()
if "Route position (km from origin)" not in st.session_state.qc_stations.columns:
    st.session_state.qc_stations["Route position (km from origin)"] = [75.0 + i * 80.0 for i in range(len(st.session_state.qc_stations))]
if "qc_slot_count" not in st.session_state:
    st.session_state.qc_slot_count = 2
if "qc_results" not in st.session_state:
    st.session_state.qc_results = None
if "qc_origin" not in st.session_state:
    st.session_state.qc_origin = "Kochi"
if "qc_destination" not in st.session_state:
    st.session_state.qc_destination = "Bengaluru"
if "qc_start_time" not in st.session_state:
    st.session_state.qc_start_time = datetime.strptime("10:00", "%H:%M").time()
if "qc_route_distance_km" not in st.session_state:
    st.session_state.qc_route_distance_km = 320.0
if "qc_average_speed_kmph" not in st.session_state:
    st.session_state.qc_average_speed_kmph = 70.0
if "qc_slot_minutes" not in st.session_state:
    st.session_state.qc_slot_minutes = 60

with st.sidebar:
    st.markdown(
        '<div style="padding:10px 4px 20px"><div style="font-size:2.4rem">⚡</div>'
        '<div style="font-size:1.42rem;font-weight:900">Quantum<span style="color:#00e5ff">Charge</span></div>'
        '<div style="font-size:.76rem;color:#8fb5dc">DYNAMIC QUANTUM EV OPTIMIZATION</div></div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "NAVIGATION",
        ["Home", "Journey Configuration", "Optimize Journey", "Results & Comparison",
         "Charging Schedule", "Quantum Details", "Scenario Data", "About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### ⚙️ Simulation Settings")
    iterations = st.slider("QAOA optimizer iterations", 5, 40, 20, 1)
    shots = st.select_slider("Quantum circuit shots", [512, 1024, 2048, 4096], value=1024)
    battery_penalty = st.slider("Battery-care preference penalty", 0, 30, 12, 1)
    st.markdown(
        '<div class="qc-note" style="margin-top:18px"><b style="color:#00e5ff">◉ LOCAL SIMULATION</b>'
        '<br><small>Synthetic journey and station data. Qiskit Aer runs locally; Statevector mode is configured up to 22 qubits. No live maps, charging networks or IBM Quantum hardware are queried.</small></div>',
        unsafe_allow_html=True,
    )

slot_count = int(st.session_state.qc_slot_count)
slots = make_slots(st.session_state.qc_start_time, slot_count, int(st.session_state.qc_slot_minutes))

if page == "Home":
    hero = hero_data()
    background = f"background-image:url('{hero}')" if hero else "background:linear-gradient(120deg,#06132b,#0b3970)"
    st.markdown(
        f'<section class="qc-hero" style="{background}"><div class="qc-hero-content">'
        '<div class="qc-eyebrow">HYBRID QUANTUM-CLASSICAL MOBILITY</div>'
        '<h1 class="qc-title">Quantum<span>Charge</span></h1>'
        '<div style="font-size:1.25rem;font-weight:700;color:#d9f5ff;margin-top:12px">Dynamic journey-aware EV charging optimization</div>'
        '<p class="qc-subtitle">Configure your journey, fleet and charging infrastructure. Explore station assignments and charging schedules using a hybrid QUBO + QAOA prototype.</p>'
        '<div class="qc-chip-row"><span class="qc-chip">DYNAMIC INPUTS</span><span class="qc-chip">QUBO FORMULATION</span><span class="qc-chip">QAOA + QISKIT</span><span class="qc-chip">CLASSICAL BENCHMARK</span></div>'
        '</div></section>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="qc-section-title">Your optimization workspace</div>', unsafe_allow_html=True)
    columns = st.columns(4)
    cards = [
        ("🧭", "Journey-aware", "Enter journey labels, route distance and average speed; position stations along the route."),
        ("🚙", "Flexible EV fleet", "Add or remove vehicles and configure battery requirements."),
        ("⚡", "Generated stations", "Create synthetic stations, chargers, capacity, tariffs and amenities from your journey inputs."),
        ("⚛️", "Hybrid optimization", "Generate a scenario-specific QUBO and compare QAOA with a classical baseline."),
    ]
    for col, card in zip(columns, cards):
        with col:
            st.markdown(f'<div class="qc-card"><div class="qc-card-icon">{card[0]}</div><div class="qc-card-title">{card[1]}</div><div class="qc-card-text">{card[2]}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="qc-section-title">Current scenario</div>', unsafe_allow_html=True)
    metrics = st.columns(3)
    metrics[0].metric("Electric vehicles", len(st.session_state.qc_evs))
    metrics[1].metric("Charging stations", len(st.session_state.qc_stations))
    metrics[2].metric("Time slots", slot_count)
    st.markdown(
        f'<div class="qc-card" style="min-height:0;margin-top:12px"><div class="qc-card-title">ACTIVE JOURNEY</div><div style="font-size:1.15rem;font-weight:800;color:#eaf8ff;overflow-wrap:anywhere">{st.session_state.qc_origin} <span style="color:#00e5ff">→</span> {st.session_state.qc_destination}</div><div class="qc-card-text">{st.session_state.qc_route_distance_km:.0f} km configured route · {st.session_state.qc_average_speed_kmph:.0f} km/h assumed average speed</div></div>',
        unsafe_allow_html=True,
    )
    st.info("Start in **Journey Configuration**. The preloaded values are editable synthetic examples, not live vehicle or station data.")

elif page == "Journey Configuration":
    page_heading("QUICK SCENARIO SETUP", "Enter your journey details", "Everything required to run the demo is collected here. Charging stations, queues, tariffs and amenities are generated automatically as a synthetic scenario.")

    st.markdown('<div class="qc-note"><b style="color:#00e5ff">ONE-STEP SETUP</b><br>Enter journey and EV details, then click Generate Scenario. The app will create a sample charging network automatically. You do not need to edit station tables.</div>', unsafe_allow_html=True)

    with st.form("qc_quick_setup_form", clear_on_submit=False):
        st.markdown("### 01 · Journey details")
        a, b = st.columns(2)
        origin_input = a.text_input("Journey origin", value=st.session_state.qc_origin)
        destination_input = b.text_input("Journey destination", value=st.session_state.qc_destination)

        c1, c2, c3 = st.columns(3)
        distance_input = c1.number_input("Total route distance (km)", min_value=1.0, max_value=5000.0, value=float(st.session_state.qc_route_distance_km), step=10.0)
        speed_input = c2.number_input("Average journey speed (km/h)", min_value=10.0, max_value=150.0, value=float(st.session_state.qc_average_speed_kmph), step=5.0)
        ev_count_input = c3.number_input(
            "Number of EVs",
            min_value=1,
            max_value=4,
            value=min(max(int(len(st.session_state.qc_evs)), 1), 4),
            step=1,
            help="Enter a whole number. The local exact-statevector QAOA prototype supports only small QUBO instances."
        )

        c4, c5, c6 = st.columns(3)
        slot_count_input = c4.number_input(
            "Number of time slots",
            min_value=2,
            max_value=3,
            value=min(max(int(st.session_state.qc_slot_count), 2), 3),
            step=1,
            help="Enter a whole number. More time slots increase the QUBO size."
        )
        slot_duration_input = c5.selectbox("Duration per slot (minutes)", [30, 45, 60, 90], index=[30,45,60,90].index(int(st.session_state.qc_slot_minutes)))
        start_time_input = c6.time_input("First slot starts at", value=st.session_state.qc_start_time)

        st.markdown("### 02 · Electric vehicle details")
        st.caption("Enter battery capacity and state of charge for each EV. Required energy is calculated automatically.")
        ev_inputs = []
        for idx in range(ev_count_input):
            existing = st.session_state.qc_evs.iloc[idx].to_dict() if idx < len(st.session_state.qc_evs) else {
                "Battery capacity (kWh)": 60.0, "Current SOC (%)": 30.0,
                "Target SOC (%)": 80.0, "Recent DC fast-charge streak": 0
            }
            st.markdown(f"**EV-{idx+1:02d}**")
            x1, x2, x3, x4 = st.columns(4)
            cap = x1.number_input(f"Battery capacity · EV-{idx+1:02d} (kWh)", min_value=10.0, max_value=250.0, value=float(existing["Battery capacity (kWh)"]), step=5.0, key=f"quick_cap_{idx}")
            soc = x2.number_input(f"Current SOC · EV-{idx+1:02d} (%)", min_value=0.0, max_value=99.0, value=float(existing["Current SOC (%)"]), step=1.0, key=f"quick_soc_{idx}")
            target = x3.number_input(f"Target SOC · EV-{idx+1:02d} (%)", min_value=1.0, max_value=100.0, value=float(existing["Target SOC (%)"]), step=1.0, key=f"quick_target_{idx}")
            streak = x4.number_input(f"Recent DC fast charges · EV-{idx+1:02d}", min_value=0, max_value=20, value=int(existing["Recent DC fast-charge streak"]), step=1, key=f"quick_streak_{idx}")
            ev_inputs.append([f"EV-{idx+1:02d}", cap, soc, target, streak])
            if target > soc:
                st.caption(f"Energy needed: **{cap * (target-soc)/100:.1f} kWh**")
            else:
                st.error(f"EV-{idx+1:02d}: Target SOC must be higher than current SOC.")

        st.markdown("### 03 · Generate charging network")
        station_count_input = st.selectbox("Number of simulated charging stations", [2, 3, 4], index=1)
        st.caption("Station locations, charger power, capacity, amenities, tariffs and queue estimates will be generated as synthetic demo data along your route.")

        submitted = st.form_submit_button("⚡ GENERATE SCENARIO & SAVE INPUTS", type="primary", width="stretch")

    if submitted:
        if not origin_input.strip() or not destination_input.strip():
            st.error("Please enter both origin and destination.")
        elif any(row[3] <= row[2] for row in ev_inputs):
            st.error("Every EV target SOC must be higher than its current SOC.")
        else:
            st.session_state.qc_origin = origin_input.strip()
            st.session_state.qc_destination = destination_input.strip()
            st.session_state.qc_route_distance_km = float(distance_input)
            st.session_state.qc_average_speed_kmph = float(speed_input)
            st.session_state.qc_slot_count = int(slot_count_input)
            st.session_state.qc_slot_minutes = int(slot_duration_input)
            st.session_state.qc_start_time = start_time_input
            st.session_state.qc_evs = pd.DataFrame(ev_inputs, columns=["EV ID", "Battery capacity (kWh)", "Current SOC (%)", "Target SOC (%)", "Recent DC fast-charge streak"])
            generated = default_stations(slot_count_input, station_count_input)
            # Spread synthetic stations along the configured route.
            positions = np.linspace(distance_input * 0.15, distance_input * 0.85, station_count_input)
            generated["Route position (km from origin)"] = positions
            generated["Capacity"] = [1 + (i % 2) for i in range(station_count_input)]
            generated["Amenities (1-5)"] = [3, 5, 4, 3][:station_count_input]
            generated["Detour (min)"] = [5, 10, 8, 12][:station_count_input]
            st.session_state.qc_stations = generated
            st.session_state.qc_results = None
            st.success("Scenario generated and saved. Synthetic stations, tariffs, queues and amenities are ready.")
            st.info("Next: open Optimize Journey, review the generated scenario and run QUBO + QAOA.")
            st.rerun()

    if len(st.session_state.qc_evs) > 0 and len(st.session_state.qc_stations) > 0:
        st.markdown("### Current saved scenario")
        summary = st.columns(4)
        summary[0].metric("EVs", len(st.session_state.qc_evs))
        summary[1].metric("Stations", len(st.session_state.qc_stations))
        summary[2].metric("Time slots", st.session_state.qc_slot_count)
        summary[3].metric("Journey distance", f"{st.session_state.qc_route_distance_km:.0f} km")
        st.caption(f"{st.session_state.qc_origin} → {st.session_state.qc_destination}. Station and tariff details are generated automatically.")

elif page == "Optimize Journey":
    page_heading("OPTIMIZATION LAB", "Review and run your scenario", "The candidate dataset and QUBO are generated from the current inputs.")
    try:
        ev_df, station_df = validate_inputs(st.session_state.qc_evs, st.session_state.qc_stations, slot_count, st.session_state.qc_route_distance_km)
        options = build_options(ev_df, station_df, slots, battery_penalty, st.session_state.qc_route_distance_km, st.session_state.qc_average_speed_kmph)
        qubo = build_qubo(options, station_df)
        metrics = st.columns(4)
        metrics[0].metric("Vehicles", len(ev_df))
        metrics[1].metric("Stations", len(station_df))
        metrics[2].metric("Decision variables", len(options))
        metrics[3].metric("Total QUBO qubits", qubo.n)
        st.markdown(f"**Journey:** {st.session_state.qc_origin} → {st.session_state.qc_destination} · {st.session_state.qc_route_distance_km:.0f} km")
        st.markdown(f"**Time windows:** " + " · ".join(slots) + f" · Assumed average speed: {st.session_state.qc_average_speed_kmph:.0f} km/h")
        with st.expander("Calculated EV energy requirements", expanded=True):
            display_evs = ev_df.copy()
            display_evs["Energy required (kWh)"] = (display_evs["Battery capacity (kWh)"] * (display_evs["Target SOC (%)"] - display_evs["Current SOC (%)"]) / 100).round(2)
            st.dataframe(display_evs, width="stretch", hide_index=True)
        with st.expander("Charging station configuration", expanded=True):
            st.dataframe(station_df, width="stretch", hide_index=True)
        with st.expander("Candidate EV / station / time-slot decisions"):
            st.dataframe(options.round(2), width="stretch", hide_index=True)
        if qubo.n > MAX_QUBITS:
            st.error(
                f"Optimization is unavailable for the current scenario: {qubo.n} QUBO qubits exceed "
                f"the local Statevector limit of {MAX_QUBITS}."
            )
            st.info(
                "This scenario is above the configured local Statevector threshold. "
                "Your entered journey, EV and station values have not been changed. "
                "Reduce the scenario dimensions only if you choose to; no automatic "
                "demo data will be substituted."
            )
        else:
            st.success(f"Scenario fits the local simulator guardrail: {qubo.n}/{MAX_QUBITS} qubits. The Run button is enabled.")
        if st.button("⚡ RUN DYNAMIC QUBO + QAOA OPTIMIZATION", type="primary", width="stretch", disabled=qubo.n > MAX_QUBITS):
            with st.spinner("Generating QUBO, solving the classical baseline, optimizing QAOA and sampling with Aer..."):
                try:
                    st.session_state.qc_results = run_scenario(ev_df, station_df, slots, iterations, shots, battery_penalty, st.session_state.qc_route_distance_km, st.session_state.qc_average_speed_kmph)
                    st.success("Optimization completed. Open Results & Comparison and Charging Schedule.")
                except Exception as exc:
                    st.error(f"Optimization could not complete: {exc}")
    except Exception as exc:
        st.error(str(exc))
        st.info("Correct the EV or station values in Journey Configuration, then return here.")

elif page == "Results & Comparison":
    page_heading("PERFORMANCE ANALYTICS", "Results & classical comparison", "Compare the best feasible QAOA measurement with the exact classical optimum.")
    results = st.session_state.qc_results
    if results is None:
        st.info("No optimization run is available. Configure a scenario and run Optimize Journey first.")
    else:
        qaoa = results["qaoa"]
        q_score = float(results["qaoa_selected"]["score"].sum())
        c_score = float(results["classical_selected"]["score"].sum())
        metrics = st.columns(4)
        metrics[0].metric("QAOA schedule score", f"{q_score:.2f}")
        metrics[1].metric("Classical optimum", f"{c_score:.2f}")
        metrics[2].metric("Feasible QAOA shots", f"{qaoa['valid_sample_fraction']*100:.2f}%")
        metrics[3].metric("QAOA qubits", results["qubo"].n)
        chart_df = pd.DataFrame({"Method": ["QAOA sampled schedule", "Exact classical optimum"], "Generalized score": [q_score, c_score]})
        chart = px.bar(chart_df, x="Method", y="Generalized score", color="Method", text_auto=".2f", color_discrete_sequence=["#00dfff", "#367cff"], template="plotly_dark")
        chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#eaf6ff", showlegend=False, yaxis_title="Generalized score (lower is better)", xaxis_title="")
        st.plotly_chart(chart, width="stretch")

        # Explain the selected schedule score in the same weighted components
        # used by build_options(). Keep raw operational values visible too.
        st.markdown("### Why this schedule was selected")
        st.write(
            "Each EV is assigned to one station and time slot. The optimizer minimizes "
            "the combined weighted score across the fleet while respecting station capacity. "
            "Lower scores are preferred."
        )
        st.latex(
            r"c_{v,s,t}=0.35T_a+D+C+1.25W+0.025E_c+A+B"
        )
        st.caption(
            "Tₐ = approach travel minutes; D = detour minutes; C = charging minutes; "
            "W = queue minutes; E꜀ = energy cost in ₹; A = amenity dissatisfaction; "
            "B = battery-care penalty."
        )

        selected = results["qaoa_selected"].copy()
        if selected.empty:
            st.warning("There is no selected assignment to break down.")
        else:
            selected["Approach contribution"] = selected["approach_min"] * 0.35
            selected["Detour contribution"] = selected["detour_min"]
            selected["Charging contribution"] = selected["charge_min"]
            selected["Queue contribution"] = selected["wait_min"] * 1.25
            selected["Energy contribution"] = selected["energy_cost"] * 0.025
            selected["Amenity contribution"] = (5 - selected["amenity_score"]) * 3.0
            selected["Battery-care contribution"] = selected["battery_penalty"]
            contribution_cols = [
                "Approach contribution", "Detour contribution",
                "Charging contribution", "Queue contribution",
                "Energy contribution", "Amenity contribution",
                "Battery-care contribution",
            ]
            selected["Recalculated score"] = selected[contribution_cols].sum(axis=1)

            breakdown_cols = [
                "ev_id", "station", "slot", *contribution_cols, "Recalculated score"
            ]
            breakdown_labels = {
                "ev_id": "Vehicle", "station": "Selected station", "slot": "Time slot",
                "Approach contribution": "Travel × 0.35",
                "Detour contribution": "Detour",
                "Charging contribution": "Charging",
                "Queue contribution": "Queue × 1.25",
                "Energy contribution": "Energy cost × 0.025",
                "Amenity contribution": "Amenity penalty",
                "Battery-care contribution": "Battery-care penalty",
                "Recalculated score": "Total score",
            }
            st.dataframe(
                selected[breakdown_cols].rename(columns=breakdown_labels).round(2),
                width="stretch", hide_index=True
            )
            st.metric("Sum of displayed EV contributions", f"{selected['Recalculated score'].sum():.2f}")
            st.caption(
                "The component values are weighted score contributions, not all physical units. "
                "For example, energy cost is in rupees before its 0.025 weighting."
            )

            st.markdown("### Assignment rationale by vehicle")
            all_options = results["options"].copy()
            for _, chosen in selected.iterrows():
                with st.expander(
                    f"{chosen['ev_id']} → {chosen['station']} · {chosen['slot']} · "
                    f"score {chosen['score']:.2f}",
                    expanded=True
                ):
                    st.write(
                        f"**Selected option:** {chosen['station']} ({chosen['charger']}) in "
                        f"{chosen['slot']}. It requires {chosen['energy_kwh']:.1f} kWh, "
                        f"has an estimated {chosen['charge_min']:.1f}-minute charge, "
                        f"{chosen['wait_min']:.0f}-minute queue and ₹{chosen['energy_cost']:.2f} "
                        f"energy cost. Amenities are rated {chosen['amenity_score']}/5."
                    )
                    vehicle_options = all_options[all_options["ev_id"] == chosen["ev_id"]].copy()
                    vehicle_options = vehicle_options.sort_values("score").head(4)
                    vehicle_options["Candidate score"] = vehicle_options["score"].round(2)
                    compare_cols = ["station", "charger", "slot", "wait_min", "charge_min", "energy_cost", "amenity_score", "Candidate score"]
                    compare_labels = {
                        "station": "Station", "charger": "Charger", "slot": "Time slot",
                        "wait_min": "Queue (min)", "charge_min": "Charge (min)",
                        "energy_cost": "Energy cost (₹)", "amenity_score": "Amenities / 5",
                    }
                    st.write("**Lowest-scoring standalone options for this EV:**")
                    st.dataframe(
                        vehicle_options[compare_cols].rename(columns=compare_labels).round(2),
                        width="stretch", hide_index=True
                    )
                    st.caption(
                        "Important: these are individual candidate rankings. The final fleet "
                        "assignment is chosen jointly, because two vehicles cannot exceed a "
                        "station's capacity in the same time slot."
                    )

            st.markdown("### Station capacity check")
            occupancy = selected.groupby(["station_id", "station", "slot", "capacity"]).size().reset_index(name="Assigned EVs")
            occupancy["Capacity check"] = occupancy.apply(
                lambda r: "Within capacity" if r["Assigned EVs"] <= r["capacity"] else "OVER CAPACITY",
                axis=1
            )
            st.dataframe(occupancy, width="stretch", hide_index=True)
            if (occupancy["Assigned EVs"] <= occupancy["capacity"]).all():
                st.success("All displayed station/time-slot assignments respect the configured capacity.")
            else:
                st.error("At least one displayed station/time-slot assignment exceeds its configured capacity.")

        st.markdown("### How to interpret the comparison")
        st.write(
            "The QAOA score is the score of the best feasible assignment found in its measurement "
            "samples (unless the app explicitly reports a classical fallback). The classical score "
            "is the exact optimum for this small configured instance. Matching scores means QAOA "
            "found an assignment with the same objective value in this run; it does not establish "
            "quantum advantage or better performance than the classical solver."
        )

        st.markdown("### QAOA run diagnostics")
        st.json({
            "gamma": round(qaoa["gamma"], 6), "beta": round(qaoa["beta"], 6),
            "expected QUBO energy": round(qaoa["expectation"], 4),
            "best feasible sampled energy": round(qaoa["energy"], 4),
            "feasible sample fraction": round(qaoa["valid_sample_fraction"], 6),
            "shots": qaoa["shots"], "simulator": "Qiskit AerSimulator",
            "optimizer": "SciPy COBYLA", "optimizer success": qaoa["optimizer_success"],
        })
        if qaoa["used_fallback"]:
            st.warning("No feasible assignment appeared in QAOA samples. The displayed schedule uses the exact classical fallback; do not describe it as a QAOA-discovered solution.")
        else:
            st.success("A feasible assignment was found in QAOA measurement samples.")
        st.caption("The classical result is exact for this configured small instance. QAOA is heuristic; no quantum advantage is claimed.")

elif page == "Charging Schedule":
    page_heading("JOURNEY PLAN", "Recommended charging schedule", "Inspect station, time, charging duration, queue, tariff, amenities and battery-care details.")
    results = st.session_state.qc_results
    if results is None:
        st.info("Run an optimization first.")
    else:
        schedule = results["qaoa_selected"].copy()
        cols = ["ev_id", "station", "charger", "slot", "route_position_km", "approach_min", "detour_min", "charge_min", "wait_min", "energy_kwh", "energy_cost", "amenity_score", "battery_penalty"]
        labels = {"ev_id":"Vehicle", "station":"Recommended station", "charger":"Charger", "slot":"Time slot", "route_position_km":"Route position (km)", "approach_min":"Approach travel (min)", "detour_min":"Detour (min)", "charge_min":"Estimated charge (min)", "wait_min":"Estimated wait (min)", "energy_kwh":"Energy (kWh)", "energy_cost":"Energy cost (₹)", "amenity_score":"Amenities (1-5)", "battery_penalty":"Battery-care penalty"}
        st.dataframe(schedule[cols].rename(columns=labels).round(2), width="stretch", hide_index=True)
        for _, row in schedule.iterrows():
            st.markdown(f"**{row['ev_id']} → {row['station']}** · {row['slot']}")
            st.caption(f"{row['charger']} · {row['charge_min']:.1f} min charging · {row['wait_min']:.0f} min estimated wait · ₹{row['energy_cost']:.2f} energy cost · amenities {row['amenity_score']}/5")

elif page == "Quantum Details":
    page_heading("QUANTUM ENGINE", "How the dynamic QUBO + QAOA model works", "Inspect the model generated for the active scenario.")
    results = st.session_state.qc_results
    if results is None:
        st.info("Run an optimization to inspect the generated model.")
    else:
        qubo = results["qubo"]
        st.markdown("### Binary decision variable")
        st.latex(r"x_{v,s,t}\in\{0,1\}")
        st.write("A decision bit is 1 when vehicle v selects station s in time slot t.")
        st.markdown("### Objective")
        st.latex(r"\min E=\sum_{v,s,t} c_{v,s,t}x_{v,s,t}")
        st.write("Candidate costs combine estimated charging duration, queue, detour, energy cost, amenity dissatisfaction and the configured battery-care preference.")
        st.markdown("### Constraints")
        st.latex(r"\sum_{s,t}x_{v,s,t}=1\quad \forall v")
        st.write("Each EV receives exactly one charging assignment.")
        st.latex(r"\sum_v x_{v,s,t}\leq Capacity_s\quad \forall s,t")
        st.write("Each station/time slot respects its configured charger capacity. Binary slack variables convert this inequality into a QUBO equality penalty.")
        metrics = st.columns(3)
        metrics[0].metric("Decision variables", qubo.decision_count)
        metrics[1].metric("Total qubits", qubo.n)
        metrics[2].metric("QAOA depth", "p = 1")
        st.markdown("### QAOA workflow")
        st.code("Initialize superposition\n→ Apply QUBO cost unitary U_C(γ)\n→ Apply X mixer U_B(β)\n→ Optimize γ and β with SciPy COBYLA\n→ Sample with Qiskit Aer\n→ Decode assignments and check feasibility")
        st.caption("Statevector evaluates expected energy during parameter optimization; AerSimulator performs shot-based sampling. This is local simulation, not quantum hardware execution.")

elif page == "Scenario Data":
    page_heading("DATA EXPLORER", "Current synthetic scenario", "Review the active journey, EVs, stations and candidate charging decisions.")
    st.markdown(f"**Journey:** {st.session_state.qc_origin} → {st.session_state.qc_destination}  ·  **Distance:** {st.session_state.get(chr(113)+chr(99)+chr(95)+chr(114)+chr(111)+chr(117)+chr(116)+chr(101)+chr(95)+chr(100)+chr(105)+chr(115)+chr(116)+chr(97)+chr(110)+chr(99)+chr(101), 650.0):.0f} km")
    st.markdown("### EV inputs")
    st.dataframe(st.session_state.qc_evs, width="stretch", hide_index=True)
    st.markdown("### Charging station inputs")
    st.dataframe(st.session_state.qc_stations, width="stretch", hide_index=True)
    if st.session_state.qc_results is not None:
        st.markdown("### Candidate charging options")
        st.dataframe(st.session_state.qc_results["options"].round(2), width="stretch", hide_index=True)

elif page == "About":
    page_heading("PROJECT OVERVIEW", "About QuantumCharge", "A hackathon prototype for journey-aware EV charging allocation.")
    st.markdown("""
    **Problem:** EV drivers may face queues, high tariffs, inconvenient charging stops and uncertain charging durations.

    **Approach:** Classical preprocessing estimates candidate charging costs. A QUBO model encodes EV-to-station/time assignments and station capacity. QAOA explores candidate bitstrings, and an exact classical baseline validates small scenarios.

    **Current scope:** User-entered origin/destination labels; configurable route distance and average speed; editable station route positions; editable synthetic EV and station data; configurable time slots; editable charger capacity; local Qiskit Aer simulation.

    **Not currently implemented:** Live maps, route distance, traffic, real station discovery, real-time queues, OEM battery charging curves, vehicle-specific battery-health prediction or live quantum hardware.

    **Battery-care note:** The fast-charge streak penalty is an illustrative user preference, not a universal battery rule. No universal “fifth charge must be AC” rule is assumed.
    """)
    st.markdown('<div class="qc-note">All scenario values are synthetic or user-supplied. Do not interpret the schedule as a real-world route or charging recommendation.</div>', unsafe_allow_html=True)

st.markdown('<div class="qc-footer">QUANTUMCHARGE · DYNAMIC QUBO · QAOA + QISKIT AER · SYNTHETIC SCENARIO</div>', unsafe_allow_html=True)
