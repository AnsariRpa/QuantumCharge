# QuantumCharge

Journey-aware EV charging optimization using a hybrid classical–quantum workflow.

## Current prototype

- Synthetic scenario: 2 EVs, 3 charging stations, 2 time slots.
- Classical preprocessing estimates charging duration, queue delay, energy cost, detour and amenity penalty.
- QUBO encodes exactly one charging option per EV and station/time capacity conflicts.
- A hand-built, one-layer QAOA circuit is optimized using SciPy.
- Qiskit `Statevector` evaluates the QAOA objective; `Qiskit AerSimulator` samples the optimized circuit.
- Exact feasible enumeration provides a classical baseline.

## Setup (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks environment activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Model

Decision variable `x[i]` indicates whether an EV uses a station/time option.

The QUBO objective combines:
- charging duration (including a simplified charging-efficiency/taper factor),
- queue delay,
- detour time,
- electricity cost,
- amenity penalty,
- a configurable-style battery-care penalty based on fast-charge streak.

Penalty terms enforce:
- exactly one option per EV;
- no two EVs assigned to the same station in the same time slot.

## Important limitations

- All input data is synthetic.
- Charging duration is a simplified estimate, not an OEM battery charging curve.
- Battery-care penalty is illustrative, not a universal fast-charge limit or battery-health prediction.
- Queue times, tariffs, amenities and travel detours are assumed inputs.
- This is a small proof-of-concept QUBO, not a production routing or charging platform.
- QAOA is heuristic. The exact classical solver is used as a baseline. No quantum advantage is claimed.
- If QAOA sampling yields no feasible bitstring, the app transparently falls back to the exact feasible solution and labels that fallback.

## Suggested next improvements

1. Add route-aware station reachability and battery energy constraints.
2. Replace synthetic data with a documented public dataset or user-provided station data.
3. Add a multi-EV scheduling horizon with charger power-sharing constraints.
4. Add configurable user priorities and richer battery-care constraints.
5. Compare QAOA against exact enumeration over multiple generated instances.

## Quantum implementation

- Qiskit SDK: circuit construction and statevector simulation.
- Qiskit Aer: shot-based circuit simulation with `AerSimulator`.
- QAOA: one cost layer and one mixer layer, with parameters optimized by SciPy COBYLA.
- Backend: local `AerSimulator` (no cloud quantum hardware is used).
