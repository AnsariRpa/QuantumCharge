"""
QuantumCharge — Journey-aware EV charging optimization.

Hybrid workflow:
- Classical preprocessing estimates charging time, queue, detour, tariff and amenity score.
- QUBO encodes one station/time choice per EV and station-time capacity conflicts.
- One-layer QAOA is optimized with SciPy and evaluated with Qiskit Statevector.
- Qiskit Aer samples the optimized circuit.
- Exact feasible enumeration provides the classical optimum for this small instance.

All scenario data is synthetic. This is a hackathon prototype, not a live charging service.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize


# =============================================================================
# PAGE CONFIGURATION AND THEME
# =============================================================================

st.set_page_config(
    page_title="QuantumCharge | Intelligent EV Charging",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT_DIR = Path(__file__).resolve().parent
HERO_IMAGE = ROOT_DIR / "assets" / "quantumcharge_hero.png"

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --qc-bg: #050b18;
    --qc-panel: #0a1730;
    --qc-panel-2: #0d2040;
    --qc-blue: #168cff;
    --qc-cyan: #00e5ff;
    --qc-text: #edf6ff;
    --qc-muted: #91a8c7;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 12% 0%, rgba(0, 116, 255, 0.16), transparent 34%),
        radial-gradient(ellipse at 92% 14%, rgba(0, 229, 255, 0.10), transparent 28%),
        linear-gradient(145deg, #040916 0%, #071329 48%, #040a17 100%);
    color: var(--qc-text);
}

[data-testid="stHeader"] {
    background: rgba(4, 9, 22, 0.75);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #06132a 0%, #071a36 55%, #040d1d 100%);
    border-right: 1px solid rgba(0, 190, 255, 0.28);
}

[data-testid="stSidebar"] * {
    color: #e8f5ff;
}

.block-container {
    padding-top: 1.6rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

h1, h2, h3 {
    color: #f4f9ff !important;
    letter-spacing: -0.035em;
}

p, li, label, .stMarkdown {
    color: #d4e5fa;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(13, 34, 67, 0.95), rgba(7, 20, 42, 0.95));
    border: 1px solid rgba(0, 190, 255, 0.25);
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 0 22px rgba(0, 120, 255, 0.08);
}

div[data-testid="stMetric"] label {
    color: #8fb9df !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #f2fbff !important;
}

.stButton > button {
    border: 1px solid #00dfff;
    border-radius: 12px;
    color: #00101e;
    background: linear-gradient(100deg, #00dfff, #168cff);
    font-weight: 800;
    min-height: 3rem;
    box-shadow: 0 0 20px rgba(0, 205, 255, 0.30);
    transition: all 0.2s ease-in-out;
}

.stButton > button:hover {
    border-color: #8af4ff;
    box-shadow: 0 0 30px rgba(0, 205, 255, 0.52);
    transform: translateY(-1px);
    color: #00101e;
}

.stSlider [data-baseweb="slider"] > div > div {
    background-color: #00dfff;
}

div[data-testid="stExpander"] {
    border: 1px solid rgba(0, 190, 255, 0.24);
    border-radius: 14px;
    background: rgba(8, 24, 48, 0.70);
}

div[data-testid="stDataFrame"] {
    border: 1px solid rgba(0, 190, 255, 0.20);
    border-radius: 12px;
    overflow: hidden;
}

hr {
    border-color: rgba(0, 190, 255, 0.20);
}

.qc-eyebrow {
    color: #00e5ff;
    text-transform: uppercase;
    font-size: 0.76rem;
    font-weight: 800;
    letter-spacing: 0.18em;
    margin-bottom: 0.55rem;
}

.qc-hero {
    position: relative;
    overflow: hidden;
    min-height: 390px;
    border: 1px solid rgba(0, 204, 255, 0.42);
    border-radius: 24px;
    padding: 42px 44px;
    background-size: cover;
    background-position: center right;
    box-shadow: 0 0 38px rgba(0, 120, 255, 0.18), inset 0 0 30px rgba(0, 170, 255, 0.06);
    display: flex;
    align-items: center;
}

.qc-hero:before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, rgba(3, 10, 27, 0.98) 0%, rgba(3, 13, 34, 0.90) 38%, rgba(3, 13, 34, 0.16) 100%);
}

.qc-hero-content {
    position: relative;
    z-index: 1;
    max-width: 650px;
}

.qc-title {
    font-size: clamp(2.6rem, 5vw, 4.8rem);
    line-height: 1.02;
    font-weight: 900;
    color: #f5fbff;
    letter-spacing: -0.065em;
    margin: 0;
}

.qc-title span {
    background: linear-gradient(90deg, #ffffff, #55eaff 55%, #168cff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.qc-subtitle {
    color: #d7edff;
    font-size: 1.18rem;
    line-height: 1.65;
    max-width: 590px;
    margin-top: 18px;
}

.qc-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 25px;
}

.qc-chip {
    color: #c8f8ff;
    border: 1px solid rgba(0, 220, 255, 0.45);
    background: rgba(0, 118, 190, 0.17);
    border-radius: 999px;
    padding: 8px 13px;
    font-size: 0.78rem;
    font-weight: 700;
    box-shadow: 0 0 14px rgba(0, 190, 255, 0.10);
}

.qc-section-title {
    color: #f3f9ff;
    font-size: 1.45rem;
    font-weight: 800;
    margin: 1.5rem 0 0.9rem 0;
}

.qc-card {
    height: 100%;
    min-height: 150px;
    border: 1px solid rgba(0, 190, 255, 0.28);
    border-radius: 18px;
    padding: 20px;
    background: linear-gradient(145deg, rgba(13, 37, 72, 0.94), rgba(7, 19, 40, 0.94));
    box-shadow: 0 0 22px rgba(0, 120, 255, 0.08);
}

.qc-card-icon {
    font-size: 1.7rem;
    margin-bottom: 9px;
}

.qc-card-title {
    color: #f2fbff;
    font-size: 1.02rem;
    font-weight: 800;
    margin-bottom: 7px;
}

.qc-card-text {
    color: #a9c2df;
    font-size: 0.88rem;
    line-height: 1.55;
}

.qc-flow {
    display: flex;
    gap: 10px;
    align-items: stretch;
    flex-wrap: wrap;
}

.qc-flow-step {
    flex: 1 1 180px;
    border: 1px solid rgba(0, 190, 255, 0.25);
    background: rgba(8, 26, 53, 0.88);
    border-radius: 14px;
    padding: 17px;
}

.qc-flow-number {
    color: #00e5ff;
    font-weight: 900;
    font-size: 0.8rem;
    letter-spacing: 0.12em;
}

.qc-flow-title {
    color: #f0f8ff;
    font-weight: 800;
    margin: 7px 0 5px 0;
}

.qc-flow-text {
    color: #9eb8d6;
    font-size: 0.82rem;
    line-height: 1.5;
}

.qc-footer {
    color: #7794b8;
    font-size: 0.76rem;
    text-align: center;
    padding: 22px 0 4px 0;
}

.qc-status {
    display: inline-block;
    border: 1px solid rgba(0, 230, 255, 0.42);
    color: #8ff5ff;
    background: rgba(0, 170, 220, 0.12);
    border-radius: 999px;
    padding: 6px 11px;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.04em;
}
</style>
"""


def load_hero_as_base64() -> str:
    """Read the bundled hero image and return a base64 data string."""
    if not HERO_IMAGE.exists():
        return ""
    encoded = base64.b64encode(HERO_IMAGE.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# SYNTHETIC SCENARIO
# =============================================================================

EVS = [
    {
        "id": "EV-01",
        "battery_pct": 22,
        "battery_kwh": 60,
        "energy_needed_kwh": 28,
        "dc_fast_streak": 3,
        "route": "Kochi → Bengaluru",
    },
    {
        "id": "EV-02",
        "battery_pct": 34,
        "battery_kwh": 75,
        "energy_needed_kwh": 32,
        "dc_fast_streak": 1,
        "route": "Kochi → Bengaluru",
    },
]

STATIONS = [
    {
        "id": "ST-A",
        "name": "Highway Hub A",
        "charger": "DC Fast",
        "power_kw": 60,
        "detour_min": 5,
        "amenity_score": 2,
        "capacity": 1,
        "queue_early": 18,
        "queue_late": 8,
        "price_early": 18.0,
        "price_late": 20.0,
    },
    {
        "id": "ST-B",
        "name": "Travel Plaza B",
        "charger": "DC Fast",
        "power_kw": 90,
        "detour_min": 10,
        "amenity_score": 5,
        "capacity": 1,
        "queue_early": 5,
        "queue_late": 12,
        "price_early": 19.0,
        "price_late": 16.0,
    },
    {
        "id": "ST-C",
        "name": "Community Charger C",
        "charger": "AC",
        "power_kw": 22,
        "detour_min": 15,
        "amenity_score": 3,
        "capacity": 1,
        "queue_early": 0,
        "queue_late": 3,
        "price_early": 11.0,
        "price_late": 12.0,
    },
]

SLOTS = ["10:00–11:00", "11:00–12:00"]


def build_options() -> pd.DataFrame:
    """Build every EV × station × time-slot candidate and its generalized cost."""
    rows = []

    for ev in EVS:
        for station in STATIONS:
            for slot_idx, slot in enumerate(SLOTS):
                if slot_idx == 0:
                    queue_minutes = station["queue_early"]
                    tariff = station["price_early"]
                else:
                    queue_minutes = station["queue_late"]
                    tariff = station["price_late"]

                # Simplified effective power factor; not an OEM charging curve.
                effective_power_kw = station["power_kw"] * 0.82
                charging_minutes = (
                    ev["energy_needed_kwh"] / effective_power_kw
                ) * 60 + 5

                # Illustrative battery-care preference, not a universal fast-charge limit.
                battery_care_penalty = 0
                if station["charger"] == "DC Fast" and ev["dc_fast_streak"] >= 3:
                    battery_care_penalty = 12

                energy_cost = ev["energy_needed_kwh"] * tariff
                amenity_penalty = (5 - station["amenity_score"]) * 3

                generalized_score = (
                    station["detour_min"]
                    + charging_minutes
                    + queue_minutes * 1.25
                    + energy_cost * 0.025
                    + amenity_penalty
                    + battery_care_penalty
                )

                rows.append(
                    {
                        "ev_id": ev["id"],
                        "station_id": station["id"],
                        "station": station["name"],
                        "charger": station["charger"],
                        "slot_idx": slot_idx,
                        "slot": slot,
                        "detour_min": station["detour_min"],
                        "charge_min": round(charging_minutes, 1),
                        "wait_min": queue_minutes,
                        "energy_cost": round(energy_cost, 2),
                        "amenity_score": station["amenity_score"],
                        "battery_penalty": battery_care_penalty,
                        "score": round(generalized_score, 4),
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# QUBO MODEL
# =============================================================================

@dataclass
class Qubo:
    linear: np.ndarray
    quadratic: Dict[Tuple[int, int], float]
    constant: float
    n: int

    def energy(self, bits: np.ndarray) -> float:
        value = self.constant + float(np.dot(self.linear, bits))
        for (i, j), coefficient in self.quadratic.items():
            value += coefficient * bits[i] * bits[j]
        return float(value)

    def all_energies(self) -> np.ndarray:
        energies = np.zeros(2 ** self.n, dtype=float)
        for state in range(2 ** self.n):
            bits = np.array(
                [(state >> qubit) & 1 for qubit in range(self.n)],
                dtype=int,
            )
            energies[state] = self.energy(bits)
        return energies


def build_qubo(
    options: pd.DataFrame,
    onehot_penalty: float = 45.0,
    capacity_penalty: float = 45.0,
) -> Qubo:
    """
    Build a QUBO:
    - Exactly one option is selected for each EV.
    - At most one EV occupies a station/time slot (capacity is 1 in this demo).
    """
    number_of_variables = len(options)
    linear = options["score"].to_numpy(dtype=float).copy()
    quadratic: Dict[Tuple[int, int], float] = {}
    constant = 0.0

    for ev_id in options["ev_id"].unique():
        indices = options.index[options["ev_id"] == ev_id].tolist()
        constant += onehot_penalty

        for i in indices:
            linear[i] -= onehot_penalty

        for position, i in enumerate(indices):
            for j in indices[position + 1:]:
                key = (min(i, j), max(i, j))
                quadratic[key] = quadratic.get(key, 0.0) + 2 * onehot_penalty

    for station_id in options["station_id"].unique():
        for slot_idx in options["slot_idx"].unique():
            station_slot_indices = options.index[
                (options["station_id"] == station_id)
                & (options["slot_idx"] == slot_idx)
            ].tolist()

            ev_groups = {}
            for index in station_slot_indices:
                ev_id = options.loc[index, "ev_id"]
                ev_groups.setdefault(ev_id, []).append(index)

            ev_ids = list(ev_groups)
            for first_group in range(len(ev_ids)):
                for second_group in range(first_group + 1, len(ev_ids)):
                    for i in ev_groups[ev_ids[first_group]]:
                        for j in ev_groups[ev_ids[second_group]]:
                            key = (min(i, j), max(i, j))
                            quadratic[key] = (
                                quadratic.get(key, 0.0) + capacity_penalty
                            )

    return Qubo(
        linear=linear,
        quadratic=quadratic,
        constant=constant,
        n=number_of_variables,
    )


# =============================================================================
# CLASSICAL BASELINE AND FEASIBILITY
# =============================================================================

def is_feasible(bits: np.ndarray, options: pd.DataFrame) -> bool:
    selected_indices = np.flatnonzero(bits)

    for ev_id in options["ev_id"].unique():
        selected_for_ev = sum(
            options.loc[index, "ev_id"] == ev_id
            for index in selected_indices
        )
        if selected_for_ev != 1:
            return False

    selected_options = options.iloc[selected_indices]
    if selected_options.duplicated(["station_id", "slot_idx"]).any():
        return False

    return True


def exact_classical_solution(
    qubo: Qubo,
    options: pd.DataFrame,
) -> Tuple[np.ndarray, float]:
    """Enumerate all bitstrings and return the best feasible assignment."""
    best_energy = float("inf")
    best_bits = None

    for state in range(2 ** qubo.n):
        bits = np.array(
            [(state >> qubit) & 1 for qubit in range(qubo.n)],
            dtype=int,
        )

        if is_feasible(bits, options):
            energy = qubo.energy(bits)
            if energy < best_energy:
                best_energy = energy
                best_bits = bits

    if best_bits is None:
        raise RuntimeError("No feasible classical assignment was found.")

    return best_bits, best_energy


# =============================================================================
# QAOA CIRCUIT AND EXECUTION
# =============================================================================

def make_qaoa_circuit(
    qubo: Qubo,
    gamma: float,
    beta: float,
    measure: bool = False,
) -> QuantumCircuit:
    """Create a one-layer QAOA circuit for the QUBO cost function."""
    circuit = QuantumCircuit(qubo.n, qubo.n if measure else 0)
    circuit.h(range(qubo.n))

    # Linear cost terms: phase on |1>.
    for qubit, coefficient in enumerate(qubo.linear):
        if abs(coefficient) > 1e-12:
            circuit.p(-gamma * float(coefficient), qubit)

    # Quadratic cost terms: controlled phase on |11>.
    for (i, j), coefficient in qubo.quadratic.items():
        if abs(coefficient) > 1e-12:
            circuit.cp(-gamma * float(coefficient), i, j)

    # X mixer.
    for qubit in range(qubo.n):
        circuit.rx(2 * beta, qubit)

    if measure:
        circuit.measure(range(qubo.n), range(qubo.n))

    return circuit


def run_qaoa(
    qubo: Qubo,
    options: pd.DataFrame,
    max_iterations: int = 18,
    shots: int = 2048,
) -> dict:
    """Optimize QAOA parameters, sample with Aer, and select the best feasible sample."""
    all_energies = qubo.all_energies()

    def expectation(parameters: np.ndarray) -> float:
        gamma, beta = parameters
        circuit = make_qaoa_circuit(qubo, float(gamma), float(beta))
        state = Statevector.from_instruction(circuit)
        probabilities = state.probabilities()
        return float(np.dot(probabilities, all_energies))

    optimization_result = minimize(
        expectation,
        x0=np.array([0.035, 0.55]),
        method="COBYLA",
        options={
            "maxiter": max_iterations,
            "rhobeg": 0.15,
            "tol": 1e-3,
        },
    )

    gamma, beta = optimization_result.x
    measured_circuit = make_qaoa_circuit(
        qubo,
        float(gamma),
        float(beta),
        measure=True,
    )

    simulator = AerSimulator()
    compiled_circuit = transpile(measured_circuit, simulator)
    counts = simulator.run(
        compiled_circuit,
        shots=shots,
    ).result().get_counts()

    best_sample = None
    best_sample_energy = float("inf")
    feasible_shots = 0

    for bitstring, frequency in counts.items():
        # Qiskit displays classical bit 0 at the right; restore q0-first ordering.
        bits = np.array(
            [int(bit) for bit in bitstring.replace(" ", "")[::-1]],
            dtype=int,
        )

        if is_feasible(bits, options):
            feasible_shots += frequency
            sample_energy = qubo.energy(bits)

            if sample_energy < best_sample_energy:
                best_sample_energy = sample_energy
                best_sample = bits

    used_fallback = False
    if best_sample is None:
        best_sample, best_sample_energy = exact_classical_solution(qubo, options)
        used_fallback = True

    return {
        "bits": best_sample,
        "energy": best_sample_energy,
        "counts": counts,
        "expectation": float(optimization_result.fun),
        "gamma": float(gamma),
        "beta": float(beta),
        "valid_sample_fraction": feasible_shots / shots,
        "used_fallback": used_fallback,
        "optimizer_success": bool(optimization_result.success),
        "optimizer_message": str(optimization_result.message),
        "shots": shots,
    }


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def show_page_heading(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="qc-eyebrow">{eyebrow}</div>
        <h1 style="margin-bottom:0.25rem;">{title}</h1>
        <p style="color:#9db8d8; margin-top:0;">{description}</p>
        """,
        unsafe_allow_html=True,
    )


def show_no_results_message() -> None:
    st.info(
        "No optimization run is available yet. Open **Optimize Journey** "
        "and run the QAOA experiment first."
    )


def get_selected_frame(options: pd.DataFrame, bits: np.ndarray) -> pd.DataFrame:
    return options.iloc[np.flatnonzero(bits)].copy()


# =============================================================================
# SIDEBAR NAVIGATION AND SETTINGS
# =============================================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding:10px 4px 20px 4px;">
            <div style="font-size:2.4rem; line-height:1;">⚡</div>
            <div style="font-size:1.42rem;font-weight:900;letter-spacing:-0.04em;">
                Quantum<span style="color:#00e5ff;">Charge</span>
            </div>
            <div style="font-size:0.76rem;color:#8fb5dc;margin-top:4px;">
                QUANTUM EV OPTIMIZATION
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_page = st.radio(
        "NAVIGATION",
        [
            "Home",
            "Optimize Journey",
            "Results & Comparison",
            "Charging Schedule",
            "Quantum Details",
            "Scenario Data",
            "About",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Simulation Settings")
    max_iterations = st.slider(
        "QAOA optimizer iterations",
        min_value=5,
        max_value=40,
        value=18,
        step=1,
    )
    circuit_shots = st.select_slider(
        "Quantum circuit shots",
        options=[512, 1024, 2048, 4096],
        value=2048,
    )

    st.markdown(
        """
        <div style="
            margin-top:18px;padding:14px;border:1px solid rgba(0,210,255,.35);
            border-radius:14px;background:rgba(0,110,190,.13);">
            <div style="color:#00e5ff;font-weight:800;margin-bottom:5px;">
                ◉ LOCAL SIMULATION
            </div>
            <div style="font-size:.78rem;color:#a9c5e4;line-height:1.5;">
                Synthetic scenario data. Qiskit Aer runs locally; no live charging
                network or IBM Quantum hardware is queried.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Build the scenario and QUBO once per Streamlit rerun.
OPTIONS = build_options()
QUBO = build_qubo(OPTIONS)


# =============================================================================
# HOME
# =============================================================================

if selected_page == "Home":
    hero_data = load_hero_as_base64()
    hero_background = (
        f"background-image:url('{hero_data}');"
        if hero_data
        else "background:linear-gradient(120deg,#06132b,#0b3970);"
    )

    st.markdown(
        f"""
        <section class="qc-hero" style="{hero_background}">
            <div class="qc-hero-content">
                <div class="qc-eyebrow">HYBRID QUANTUM-CLASSICAL MOBILITY</div>
                <h1 class="qc-title">Quantum<span>Charge</span></h1>
                <div style="font-size:1.28rem;font-weight:700;color:#d9f5ff;margin-top:12px;">
                    Journey-aware EV charging optimization
                </div>
                <p class="qc-subtitle">
                    Plan smarter charging stops by balancing journey time, charging
                    duration, waiting, energy cost, station amenities and battery-care
                    preferences.
                </p>
                <div class="qc-chip-row">
                    <span class="qc-chip">QUBO FORMULATION</span>
                    <span class="qc-chip">QAOA + QISKIT</span>
                    <span class="qc-chip">CLASSICAL BENCHMARK</span>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="qc-section-title">The optimization mission</div>', unsafe_allow_html=True)
    feature_columns = st.columns(4)

    feature_cards = [
        ("⏱️", "Less waiting", "Account for station queue estimates and charging duration."),
        ("⚡", "Lower energy cost", "Compare time-dependent tariffs and energy requirements."),
        ("☕", "Better amenities", "Prefer useful stops while accounting for detour time."),
        ("🔋", "Battery-care aware", "Apply a configurable preference against repeated fast charging."),
    ]

    for column, card in zip(feature_columns, feature_cards):
        with column:
            st.markdown(
                f"""
                <div class="qc-card">
                    <div class="qc-card-icon">{card[0]}</div>
                    <div class="qc-card-title">{card[1]}</div>
                    <div class="qc-card-text">{card[2]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="qc-section-title">Prototype at a glance</div>', unsafe_allow_html=True)
    metric_columns = st.columns(4)
    metric_columns[0].metric("Electric vehicles", len(EVS))
    metric_columns[1].metric("Charging stations", len(STATIONS))
    metric_columns[2].metric("Time slots", len(SLOTS))
    metric_columns[3].metric("QUBO variables", QUBO.n)

    st.markdown('<div class="qc-section-title">How QuantumCharge works</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="qc-flow">
            <div class="qc-flow-step">
                <div class="qc-flow-number">STEP 01</div>
                <div class="qc-flow-title">Journey inputs</div>
                <div class="qc-flow-text">EV energy needs, station data, tariffs, queues and amenities.</div>
            </div>
            <div class="qc-flow-step">
                <div class="qc-flow-number">STEP 02</div>
                <div class="qc-flow-title">QUBO formulation</div>
                <div class="qc-flow-text">Translate station and time-slot decisions into binary variables.</div>
            </div>
            <div class="qc-flow-step">
                <div class="qc-flow-number">STEP 03</div>
                <div class="qc-flow-title">QAOA exploration</div>
                <div class="qc-flow-text">Optimize a parameterized quantum circuit and sample with Qiskit Aer.</div>
            </div>
            <div class="qc-flow-step">
                <div class="qc-flow-number">STEP 04</div>
                <div class="qc-flow-title">Schedule & insights</div>
                <div class="qc-flow-text">Inspect feasible assignments and compare with an exact classical baseline.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    if st.button("⚡  START JOURNEY OPTIMIZATION", width="stretch"):
        st.session_state["navigation_target"] = "Optimize Journey"
        st.info("Choose **Optimize Journey** in the sidebar to run the experiment.")

    st.markdown(
        '<div class="qc-footer">QUANTUMCHARGE · HACKATHON PROTOTYPE · SIMULATED DATA</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# OPTIMIZATION PAGE
# =============================================================================

elif selected_page == "Optimize Journey":
    show_page_heading(
        "OPTIMIZATION LAB",
        "Configure the charging scenario",
        "Review the synthetic inputs, then run the hybrid QUBO + QAOA optimization.",
    )

    metric_columns = st.columns(3)
    metric_columns[0].metric("Vehicles", len(EVS))
    metric_columns[1].metric("Stations", len(STATIONS))
    metric_columns[2].metric("Decision variables", QUBO.n)

    with st.expander("View electric vehicle inputs", expanded=True):
        vehicle_frame = pd.DataFrame(EVS).rename(
            columns={
                "id": "EV",
                "battery_pct": "Current battery (%)",
                "battery_kwh": "Battery capacity (kWh)",
                "energy_needed_kwh": "Energy required (kWh)",
                "dc_fast_streak": "Recent DC fast-charge streak",
                "route": "Journey",
            }
        )
        st.dataframe(vehicle_frame, width="stretch", hide_index=True)

    with st.expander("View charging station inputs", expanded=True):
        station_frame = pd.DataFrame(STATIONS).rename(
            columns={
                "id": "Station ID",
                "name": "Station",
                "charger": "Charger type",
                "power_kw": "Power (kW)",
                "detour_min": "Detour (min)",
                "amenity_score": "Amenities (1–5)",
                "capacity": "Capacity",
                "queue_early": "Queue 10 AM",
                "queue_late": "Queue 11 AM",
                "price_early": "Tariff 10 AM (₹/kWh)",
                "price_late": "Tariff 11 AM (₹/kWh)",
            }
        )
        st.dataframe(station_frame, width="stretch", hide_index=True)

    st.markdown('<div class="qc-section-title">Run the quantum optimization</div>', unsafe_allow_html=True)
    st.markdown(
        """
        The experiment will:
        1. Enumerate feasible assignments to establish the exact classical baseline.
        2. Build the QAOA circuit from the QUBO coefficients.
        3. Optimize QAOA parameters with SciPy COBYLA.
        4. Sample the optimized circuit on the local Qiskit Aer simulator.
        """,
    )

    if st.button("⚡ RUN QUANTUMCHARGE OPTIMIZATION", type="primary", width="stretch"):
        with st.spinner("Computing exact classical baseline..."):
            classical_bits, classical_energy = exact_classical_solution(QUBO, OPTIONS)

        with st.spinner("Optimizing QAOA parameters and sampling with Qiskit Aer..."):
            qaoa_result = run_qaoa(
                QUBO,
                OPTIONS,
                max_iterations=max_iterations,
                shots=circuit_shots,
            )

        classical_selected = get_selected_frame(OPTIONS, classical_bits)
        qaoa_selected = get_selected_frame(OPTIONS, qaoa_result["bits"])

        st.session_state["quantumcharge_results"] = {
            "classical_bits": classical_bits,
            "classical_energy": classical_energy,
            "qaoa": qaoa_result,
            "classical_selected": classical_selected,
            "qaoa_selected": qaoa_selected,
        }

        st.success("Optimization run completed. Open Results & Comparison to inspect the outcome.")


# =============================================================================
# RESULTS PAGE
# =============================================================================

elif selected_page == "Results & Comparison":
    show_page_heading(
        "PERFORMANCE ANALYTICS",
        "Results & classical comparison",
        "Compare the best feasible QAOA sample against the exact classical optimum.",
    )

    if "quantumcharge_results" not in st.session_state:
        show_no_results_message()
    else:
        results = st.session_state["quantumcharge_results"]
        qaoa_result = results["qaoa"]
        qaoa_selected = results["qaoa_selected"]
        classical_selected = results["classical_selected"]

        qaoa_score = float(qaoa_selected["score"].sum())
        classical_score = float(classical_selected["score"].sum())

        metric_columns = st.columns(4)
        metric_columns[0].metric("QAOA schedule score", f"{qaoa_score:.2f}")
        metric_columns[1].metric("Classical optimum score", f"{classical_score:.2f}")
        metric_columns[2].metric(
            "Feasible QAOA shots",
            f"{qaoa_result['valid_sample_fraction'] * 100:.1f}%",
        )
        metric_columns[3].metric("QAOA qubits", QUBO.n)

        comparison_frame = pd.DataFrame(
            {
                "Method": ["QAOA sampled schedule", "Exact classical optimum"],
                "Generalized score": [qaoa_score, classical_score],
            }
        )

        chart = px.bar(
            comparison_frame,
            x="Method",
            y="Generalized score",
            color="Method",
            text_auto=".2f",
            color_discrete_sequence=["#00dfff", "#367cff"],
            template="plotly_dark",
        )
        chart.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#eaf6ff",
            showlegend=False,
            yaxis_title="Generalized score (lower is better)",
            xaxis_title="",
        )
        st.plotly_chart(chart, width="stretch")

        st.markdown("### QAOA run diagnostics")
        diagnostics = {
            "QAOA gamma": round(qaoa_result["gamma"], 6),
            "QAOA beta": round(qaoa_result["beta"], 6),
            "Expected QUBO energy": round(qaoa_result["expectation"], 4),
            "Best feasible sampled energy": round(qaoa_result["energy"], 4),
            "Feasible sample fraction": round(qaoa_result["valid_sample_fraction"], 4),
            "Shots": qaoa_result["shots"],
            "Simulator": "Qiskit AerSimulator",
            "Optimizer": "SciPy COBYLA",
        }
        st.json(diagnostics)

        if qaoa_result["used_fallback"]:
            st.warning(
                "No feasible assignment appeared in the QAOA measurement samples. "
                "The displayed QAOA schedule is an exact-classical fallback and must "
                "not be described as a QAOA-discovered solution."
            )
        else:
            st.success("A feasible assignment was found in the QAOA measurement samples.")

        st.caption(
            "The exact classical result is the optimum for this small synthetic instance. "
            "QAOA is evaluated as a heuristic; no quantum advantage is claimed."
        )


# =============================================================================
# CHARGING SCHEDULE PAGE
# =============================================================================

elif selected_page == "Charging Schedule":
    show_page_heading(
        "JOURNEY PLAN",
        "Recommended charging schedule",
        "Inspect station, charging, queue, tariff and amenity details for each EV.",
    )

    if "quantumcharge_results" not in st.session_state:
        show_no_results_message()
    else:
        results = st.session_state["quantumcharge_results"]
        schedule = results["qaoa_selected"].copy()

        schedule_columns = [
            "ev_id",
            "station",
            "charger",
            "slot",
            "detour_min",
            "charge_min",
            "wait_min",
            "energy_cost",
            "amenity_score",
            "battery_penalty",
        ]

        schedule_display = schedule[schedule_columns].rename(
            columns={
                "ev_id": "Vehicle",
                "station": "Recommended station",
                "charger": "Charger",
                "slot": "Time slot",
                "detour_min": "Detour (min)",
                "charge_min": "Estimated charge (min)",
                "wait_min": "Estimated wait (min)",
                "energy_cost": "Energy cost (₹)",
                "amenity_score": "Amenities (1–5)",
                "battery_penalty": "Battery-care penalty",
            }
        )

        st.dataframe(schedule_display, width="stretch", hide_index=True)

        st.markdown("### Journey recommendation notes")
        for _, row in schedule.iterrows():
            st.markdown(
                f"""
                <div class="qc-card" style="min-height:unset;margin-bottom:12px;">
                    <div class="qc-card-title">{row['ev_id']} → {row['station']}</div>
                    <div class="qc-card-text">
                        {row['charger']} charger · {row['slot']} ·
                        {row['charge_min']} min estimated charging ·
                        {row['wait_min']} min estimated queue ·
                        ₹{row['energy_cost']:.2f} estimated energy cost ·
                        Amenity score {row['amenity_score']}/5
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# QUANTUM DETAILS PAGE
# =============================================================================

elif selected_page == "Quantum Details":
    show_page_heading(
        "QUANTUM ENGINE",
        "How the QUBO + QAOA model works",
        "A transparent view of the binary decision model and quantum circuit workflow.",
    )

    st.markdown("### Binary decision variable")
    st.latex(
        r"x_{v,s,t} = "
        r"\begin{cases}"
        r"1 & \text{if vehicle }v\text{ selects station }s\text{ at time }t \\\\"
        r"0 & \text{otherwise}"
        r"\end{cases}"
    )

    st.markdown("### Objective function")
    st.latex(
        r"\min E = "
        r"w_T T + w_C C + w_W W + w_D D + w_A A + w_B B"
    )

    st.markdown(
        """
        - **T:** estimated charging duration and journey-time contribution.
        - **C:** electricity cost based on the assumed tariff.
        - **W:** estimated queue delay.
        - **D:** station detour time.
        - **A:** amenity dissatisfaction penalty.
        - **B:** illustrative battery-care penalty.
        """
    )

    st.markdown("### QUBO constraints")
    st.latex(
        r"A\left(\sum_{s,t}x_{v,s,t}-1\right)^2"
    )
    st.write(
        "Each EV must select exactly one station/time option. A separate "
        "pairwise penalty prevents two EVs from occupying the same station "
        "in the same time slot in this capacity-one scenario."
    )

    st.markdown("### QAOA workflow")
    st.code(
        "Initialize superposition\n"
        "    ↓\n"
        "Apply QUBO cost unitary U_C(gamma)\n"
        "    ↓\n"
        "Apply X-mixer U_B(beta)\n"
        "    ↓\n"
        "Optimize gamma and beta with SciPy COBYLA\n"
        "    ↓\n"
        "Sample optimized circuit with Qiskit Aer\n"
        "    ↓\n"
        "Decode bitstrings and check feasibility",
        language="text",
    )

    st.markdown("### Implementation facts")
    fact_columns = st.columns(3)
    fact_columns[0].metric("Qubits", QUBO.n)
    fact_columns[1].metric("QAOA depth", "p = 1")
    fact_columns[2].metric("Backend", "AerSimulator")

    st.warning(
        "This is a one-layer QAOA prototype on a small synthetic problem. "
        "It demonstrates meaningful quantum optimization integration but "
        "does not establish quantum advantage."
    )


# =============================================================================
# SCENARIO DATA PAGE
# =============================================================================

elif selected_page == "Scenario Data":
    show_page_heading(
        "DATA EXPLORER",
        "Synthetic optimization dataset",
        "Review all candidate EV-to-station/time-slot choices and their estimated costs.",
    )

    st.markdown("### Candidate charging options")
    st.dataframe(OPTIONS, width="stretch", hide_index=True)

    st.markdown("### Generalized score by station and time")
    score_frame = (
        OPTIONS.groupby(["station", "slot"], as_index=False)["score"]
        .mean()
        .sort_values("score")
    )

    score_chart = px.bar(
        score_frame,
        x="station",
        y="score",
        color="slot",
        barmode="group",
        template="plotly_dark",
        color_discrete_sequence=["#00dfff", "#367cff"],
        text_auto=".1f",
    )
    score_chart.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#eaf6ff",
        yaxis_title="Average generalized score (lower is better)",
        xaxis_title="Charging station",
    )
    st.plotly_chart(score_chart, width="stretch")


# =============================================================================
# ABOUT PAGE
# =============================================================================

elif selected_page == "About":
    show_page_heading(
        "PROJECT OVERVIEW",
        "About QuantumCharge",
        "A hackathon prototype exploring quantum optimization for electric mobility.",
    )

    st.markdown(
        """
        <div class="qc-card" style="min-height:unset;">
            <div class="qc-card-title">The problem</div>
            <div class="qc-card-text">
                EV charging decisions can create queues, additional travel time,
                high energy costs and inconvenient stops. A charging decision should
                consider more than the nearest charger.
            </div>
            <br>
            <div class="qc-card-title">Our approach</div>
            <div class="qc-card-text">
                QuantumCharge combines classical preprocessing with a QUBO model
                and QAOA circuit to investigate station and time-slot allocation.
                It evaluates charging time, waiting, detour, tariff, amenities and
                a configurable battery-care preference.
            </div>
            <br>
            <div class="qc-card-title">Technology stack</div>
            <div class="qc-card-text">
                Python · Streamlit · Qiskit · Qiskit Aer · SciPy · NumPy · Pandas · Plotly
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Prototype limitations")
    st.warning(
        "All inputs are synthetic. Charging time uses a simplified effective-power "
        "estimate, not an OEM battery charging curve. Queue, tariff, detour and "
        "amenity values are assumptions. The model does not yet perform live route "
        "planning, live charger discovery or battery-health prediction."
    )

    st.markdown("### Hackathon alignment")
    st.write(
        "Challenge 04 — Quantum Optimization | Electric Mobility. "
        "The project uses QUBO and QAOA with Qiskit and a local simulator, "
        "and compares the heuristic quantum samples with an exact classical baseline."
    )

st.markdown(
    '<div class="qc-footer">QUANTUMCHARGE · QISKIT + QAOA · LOCAL AER SIMULATION</div>',
    unsafe_allow_html=True,
)
