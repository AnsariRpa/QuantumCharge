"""
QuantumCharge: journey-aware EV charging optimization prototype.

Hybrid workflow:
1. Classical preprocessing estimates charging time and generalized journey cost.
2. A QUBO encodes one charging choice per EV and station/time capacity.
3. A hand-built QAOA circuit is optimized with SciPy and evaluated with Qiskit Statevector.
4. Qiskit Aer samples the optimized circuit.
5. An exact classical enumeration provides a transparent baseline.

All data is synthetic and intended for a hackathon prototype.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


# ---------------------------- Synthetic scenario ----------------------------

EVS = [
    {"id": "EV-01", "battery_pct": 22, "battery_kwh": 60, "energy_needed_kwh": 28,
     "dc_fast_streak": 3, "route": "Kochi → Bengaluru"},
    {"id": "EV-02", "battery_pct": 34, "battery_kwh": 75, "energy_needed_kwh": 32,
     "dc_fast_streak": 1, "route": "Kochi → Bengaluru"},
]

STATIONS = [
    {"id": "ST-A", "name": "Highway Hub A", "charger": "DC Fast", "power_kw": 60,
     "detour_min": 5, "amenity_score": 2, "capacity": 1, "queue_early": 18,
     "queue_late": 8, "price_early": 18.0, "price_late": 20.0},
    {"id": "ST-B", "name": "Travel Plaza B", "charger": "DC Fast", "power_kw": 90,
     "detour_min": 10, "amenity_score": 5, "capacity": 1, "queue_early": 5,
     "queue_late": 12, "price_early": 19.0, "price_late": 16.0},
    {"id": "ST-C", "name": "Community Charger C", "charger": "AC", "power_kw": 22,
     "detour_min": 15, "amenity_score": 3, "capacity": 1, "queue_early": 0,
     "queue_late": 3, "price_early": 11.0, "price_late": 12.0},
]

SLOTS = ["10:00–11:00", "11:00–12:00"]


def build_options() -> pd.DataFrame:
    rows = []
    for ev in EVS:
        for station in STATIONS:
            for slot_idx, slot in enumerate(SLOTS):
                queue = station["queue_early"] if slot_idx == 0 else station["queue_late"]
                price = station["price_early"] if slot_idx == 0 else station["price_late"]

                # Effective power is reduced to represent charging taper and real-world losses.
                effective_kw = station["power_kw"] * 0.82
                charge_min = ev["energy_needed_kwh"] / effective_kw * 60
                charge_min += 5  # connection / handling overhead

                # Battery-care preference: a soft penalty, not a universal fast-charge prohibition.
                battery_penalty = 0
                if station["charger"] == "DC Fast" and ev["dc_fast_streak"] >= 3:
                    battery_penalty = 12

                # Generalized cost in weighted minutes-equivalent.
                energy_cost = ev["energy_needed_kwh"] * price
                amenity_penalty = (5 - station["amenity_score"]) * 3
                score = (
                    station["detour_min"] * 1.0
                    + charge_min * 1.0
                    + queue * 1.25
                    + energy_cost * 0.025
                    + amenity_penalty
                    + battery_penalty
                )
                rows.append({
                    "ev_id": ev["id"],
                    "station_id": station["id"],
                    "station": station["name"],
                    "charger": station["charger"],
                    "slot_idx": slot_idx,
                    "slot": slot,
                    "detour_min": station["detour_min"],
                    "charge_min": round(charge_min, 1),
                    "wait_min": queue,
                    "energy_cost": round(energy_cost, 2),
                    "amenity_score": station["amenity_score"],
                    "battery_penalty": battery_penalty,
                    "score": round(score, 4),
                })
    return pd.DataFrame(rows)


# ------------------------------- QUBO model ---------------------------------

@dataclass
class Qubo:
    linear: np.ndarray
    quadratic: Dict[Tuple[int, int], float]
    constant: float
    n: int

    def energy(self, bits: np.ndarray) -> float:
        value = self.constant + float(np.dot(self.linear, bits))
        for (i, j), coeff in self.quadratic.items():
            value += coeff * bits[i] * bits[j]
        return float(value)

    def all_energies(self) -> np.ndarray:
        energies = np.zeros(2 ** self.n, dtype=float)
        for state in range(2 ** self.n):
            bits = np.array([(state >> q) & 1 for q in range(self.n)], dtype=int)
            energies[state] = self.energy(bits)
        return energies


def build_qubo(options: pd.DataFrame, onehot_penalty: float = 45.0,
               capacity_penalty: float = 45.0) -> Qubo:
    n = len(options)
    linear = options["score"].to_numpy(dtype=float).copy()
    quadratic: Dict[Tuple[int, int], float] = {}
    constant = 0.0

    # Exactly one station/time option must be selected for each EV:
    # A * (sum(x_i) - 1)^2 = -A*sum(x_i) + 2A*sum(x_i*x_j) + constant.
    for ev_id in options["ev_id"].unique():
        indices = options.index[options["ev_id"] == ev_id].tolist()
        constant += onehot_penalty
        for i in indices:
            linear[i] -= onehot_penalty
        for pos, i in enumerate(indices):
            for j in indices[pos + 1:]:
                key = (min(i, j), max(i, j))
                quadratic[key] = quadratic.get(key, 0.0) + 2 * onehot_penalty

    # At most one EV may occupy the same station in the same time slot.
    for station_id in options["station_id"].unique():
        for slot_idx in options["slot_idx"].unique():
            indices = options.index[
                (options["station_id"] == station_id) &
                (options["slot_idx"] == slot_idx)
            ].tolist()
            ev_groups = {}
            for i in indices:
                ev_groups.setdefault(options.loc[i, "ev_id"], []).append(i)
            # Add pairwise collision penalties across different EVs only.
            ev_ids = list(ev_groups)
            for a in range(len(ev_ids)):
                for b in range(a + 1, len(ev_ids)):
                    for i in ev_groups[ev_ids[a]]:
                        for j in ev_groups[ev_ids[b]]:
                            key = (min(i, j), max(i, j))
                            quadratic[key] = quadratic.get(key, 0.0) + capacity_penalty

    return Qubo(linear, quadratic, constant, n)


# ----------------------------- Classical baseline ---------------------------

def is_feasible(bits: np.ndarray, options: pd.DataFrame) -> bool:
    selected = np.flatnonzero(bits)
    for ev_id in options["ev_id"].unique():
        if sum(options.loc[i, "ev_id"] == ev_id for i in selected) != 1:
            return False
    chosen = options.iloc[selected]
    if chosen.duplicated(["station_id", "slot_idx"]).any():
        return False
    return True


def exact_classical_solution(qubo: Qubo, options: pd.DataFrame):
    best_energy = float("inf")
    best_bits = None
    for state in range(2 ** qubo.n):
        bits = np.array([(state >> q) & 1 for q in range(qubo.n)], dtype=int)
        if is_feasible(bits, options):
            energy = qubo.energy(bits)
            if energy < best_energy:
                best_energy, best_bits = energy, bits
    return best_bits, best_energy


# ---------------------------------- QAOA -------------------------------------

def make_qaoa_circuit(qubo: Qubo, gamma: float, beta: float,
                      measure: bool = False) -> QuantumCircuit:
    qc = QuantumCircuit(qubo.n, qubo.n if measure else 0)
    qc.h(range(qubo.n))

    # Cost unitary exp(-i * gamma * Q(x)); x_i is the computational-basis bit.
    for i, coeff in enumerate(qubo.linear):
        if abs(coeff) > 1e-12:
            qc.p(-gamma * float(coeff), i)
    for (i, j), coeff in qubo.quadratic.items():
        if abs(coeff) > 1e-12:
            qc.cp(-gamma * float(coeff), i, j)

    # Standard X mixer.
    for i in range(qubo.n):
        qc.rx(2 * beta, i)

    if measure:
        qc.measure(range(qubo.n), range(qubo.n))
    return qc


def run_qaoa(qubo: Qubo, options: pd.DataFrame, maxiter: int = 18,
             shots: int = 2048):
    energies = qubo.all_energies()

    def expectation(params):
        gamma, beta = params
        circuit = make_qaoa_circuit(qubo, gamma, beta)
        state = Statevector.from_instruction(circuit)
        probabilities = state.probabilities()
        return float(np.dot(probabilities, energies))

    result = minimize(
        expectation,
        x0=np.array([0.035, 0.55]),
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 0.15, "tol": 1e-3},
    )

    gamma, beta = result.x
    measured_circuit = make_qaoa_circuit(qubo, float(gamma), float(beta), measure=True)
    backend = AerSimulator()
    compiled = transpile(measured_circuit, backend)
    counts = backend.run(compiled, shots=shots).result().get_counts()

    # Qiskit count strings are displayed most-significant classical bit first.
    # Reverse to recover our variable order q0, q1, ... .
    best_sample = None
    best_sample_energy = float("inf")
    valid_samples = 0
    for bitstring, frequency in counts.items():
        bits = np.array([int(bit) for bit in bitstring.replace(" ", "")[::-1]], dtype=int)
        if is_feasible(bits, options):
            valid_samples += frequency
            energy = qubo.energy(bits)
            if energy < best_sample_energy:
                best_sample_energy = energy
                best_sample = bits

    if best_sample is None:
        # Honest fallback: return the best feasible exact solution if QAOA samples
        # did not contain a feasible bitstring. The UI reports this explicitly.
        best_sample, best_sample_energy = exact_classical_solution(qubo, options)
        used_fallback = True
    else:
        used_fallback = False

    return {
        "bits": best_sample,
        "energy": best_sample_energy,
        "counts": counts,
        "expectation": float(result.fun),
        "gamma": float(gamma),
        "beta": float(beta),
        "valid_sample_fraction": valid_samples / shots,
        "used_fallback": used_fallback,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }


# ---------------------------------- UI ---------------------------------------

st.set_page_config(page_title="QuantumCharge", page_icon="⚡", layout="wide")

st.title("⚡ QuantumCharge")
st.subheader("Journey-aware EV charging optimization")
st.caption("Hybrid quantum-classical prototype · Synthetic hackathon scenario")

with st.sidebar:
    st.header("Optimization settings")
    maxiter = st.slider("QAOA optimizer iterations", 5, 40, 18)
    shots = st.select_slider("Quantum circuit shots", options=[512, 1024, 2048, 4096], value=2048)
    st.info("This prototype uses simulated data. It does not query live charging networks.")

options = build_options()
qubo = build_qubo(options)

c1, c2, c3 = st.columns(3)
c1.metric("Electric vehicles", len(EVS))
c2.metric("Charging stations", len(STATIONS))
c3.metric("Decision variables", qubo.n)

with st.expander("Scenario inputs", expanded=False):
    st.write("**Vehicles**")
    st.dataframe(pd.DataFrame(EVS), use_container_width=True, hide_index=True)
    st.write("**Charging stations**")
    st.dataframe(pd.DataFrame(STATIONS), use_container_width=True, hide_index=True)

if st.button("Run QuantumCharge optimization", type="primary", use_container_width=True):
    with st.spinner("Solving exact classical baseline..."):
        classical_bits, classical_energy = exact_classical_solution(qubo, options)

    with st.spinner("Building QAOA circuit, optimizing parameters and sampling with Qiskit Aer..."):
        qaoa = run_qaoa(qubo, options, maxiter=maxiter, shots=shots)

    classical_selected = options.iloc[np.flatnonzero(classical_bits)].copy()
    quantum_selected = options.iloc[np.flatnonzero(qaoa["bits"])].copy()

    st.session_state["results"] = {
        "classical_bits": classical_bits,
        "classical_energy": classical_energy,
        "qaoa": qaoa,
        "classical_selected": classical_selected,
        "quantum_selected": quantum_selected,
    }

if "results" in st.session_state:
    results = st.session_state["results"]
    qaoa = results["qaoa"]
    quantum_selected = results["quantum_selected"]
    classical_selected = results["classical_selected"]

    st.divider()
    st.header("Recommended charging schedule")

    display_cols = ["ev_id", "station", "charger", "slot", "detour_min",
                    "charge_min", "wait_min", "energy_cost", "amenity_score"]
    st.dataframe(
        quantum_selected[display_cols].rename(columns={
            "ev_id": "EV", "station": "Recommended station", "charger": "Charger",
            "slot": "Time slot", "detour_min": "Detour (min)",
            "charge_min": "Charge (min)", "wait_min": "Wait (min)",
            "energy_cost": "Energy cost (₹)", "amenity_score": "Amenities / 5",
        }),
        use_container_width=True, hide_index=True
    )

    total_q = float(quantum_selected["score"].sum())
    total_c = float(classical_selected["score"].sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("QAOA selected objective", f"{total_q:.2f}")
    m2.metric("Exact classical objective", f"{total_c:.2f}")
    m3.metric("Feasible QAOA samples", f"{qaoa['valid_sample_fraction']*100:.1f}%")
    m4.metric("QAOA circuit", f"{qubo.n} qubits")

    st.subheader("Quantum vs classical assignment")
    comparison = pd.DataFrame({
        "Method": ["QAOA", "Exact classical"],
        "Objective (lower is better)": [total_q, total_c],
    })
    st.bar_chart(comparison.set_index("Method"))

    if qaoa["used_fallback"]:
        st.warning(
            "No feasible assignment appeared in the QAOA measurement samples. "
            "The displayed schedule uses the exact classical feasible solution as a fallback. "
            "Do not present this fallback as a QAOA-discovered solution."
        )
    else:
        st.success("A feasible schedule was found among the QAOA measurement samples.")

    st.subheader("QAOA run details")
    st.json({
        "gamma": round(qaoa["gamma"], 5),
        "beta": round(qaoa["beta"], 5),
        "expected_QUBO_energy": round(qaoa["expectation"], 4),
        "best_feasible_sample_energy": round(qaoa["energy"], 4),
        "valid_sample_fraction": round(qaoa["valid_sample_fraction"], 4),
        "optimizer_message": qaoa["optimizer_message"],
        "simulator": "Qiskit AerSimulator",
    })

    st.caption(
        "Interpretation: QAOA is evaluated as a heuristic on this small synthetic instance. "
        "The exact classical solver is the optimal baseline; no quantum advantage is claimed."
    )
else:
    st.info("Click **Run QuantumCharge optimization** to generate the schedule and comparison.")
    st.write("The prototype currently models one charging decision per EV across three stations and two time slots.")
