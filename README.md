# ⚡ QuantumCharge 2.0

**Dynamic journey-aware EV charging optimization using a hybrid classical–quantum workflow.**

QuantumCharge lets users configure journey labels, EV fleet data, charging stations, time windows, station capacity, tariffs, queues and amenities. The application builds a scenario-specific candidate dataset and QUBO, explores it with QAOA, and compares the result with an exact classical baseline for small instances.

## What's new in 2.1

- Full journey text display without metric-card truncation.
- Configurable total route distance and assumed average speed.
- Editable station position along the route.
- Route approach time contributes to each candidate assignment score.
- Live scenario preview and explicit Apply & Save validation.
- Saving a changed scenario clears prior optimization results to avoid stale comparisons.

## What's new in 2.0

- User-entered journey origin and destination labels.
- Editable EV fleet with add/remove rows.
- Battery capacity, current SOC, target SOC and recent DC fast-charge streak.
- Energy requirement calculated from battery capacity and SOC target.
- Editable synthetic charging stations with charger type, power, detour, amenities and capacity.
- Configurable time slots with per-slot tariff and queue estimates.
- Capacity constraints represented with binary slack variables in the QUBO.
- Scenario-dependent decision-variable and qubit counts.
- Guardrails for local Statevector and exact classical computation.
- Existing dark navy / neon cyan interface retained.

## Location inputs

Origin and destination are labels in this version. They do not geocode locations, calculate road distances, identify real charging stations, or retrieve traffic data. Station examples are synthetic and editable.

## Setup — Windows PowerShell

Python 3.12 is recommended.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If your virtual environment already exists, activate it and run:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

## Quick setup workflow

1. Open **Journey Configuration**.
2. Enter origin, destination, route distance, average speed and time-slot settings.
3. Enter each EV's battery capacity, current SOC, target SOC and recent DC fast-charge streak.
4. Choose how many synthetic stations to generate.
5. Click **Generate Scenario & Save Inputs**. Station positions, capacity, queues, tariffs and amenities are generated automatically.
6. Open **Optimize Journey**, review the scenario and run optimization.

## Workflow

1. Open **Journey Configuration**.
2. Enter origin and destination labels.
3. Select 2 or 3 time slots and configure their start/duration.
4. Edit the EV table; add/remove EV rows.
5. Edit synthetic charging stations; add/remove stations and configure per-slot queue and tariff.
6. Open **Optimize Journey** and review calculated energy requirements and candidate options.
7. Run **Dynamic QUBO + QAOA Optimization**.
8. Inspect **Results & Comparison**, **Charging Schedule**, and **Quantum Details**.

## Quantum implementation

- Qiskit constructs the quantum circuit.
- Qiskit Statevector evaluates expected QUBO energy during parameter optimization.
- Qiskit Aer performs shot-based sampling.
- QAOA uses one cost layer and one X-mixer layer (p = 1).
- SciPy COBYLA optimizes gamma and beta.
- The classical baseline enumerates one candidate choice per EV and checks station/time capacity.

The current implementation is a local simulation. It does not execute on IBM Quantum hardware and does not claim quantum advantage.

## Model

Binary decision variable `x(v,s,t) = 1` when vehicle `v` selects station `s` in time slot `t`.

Constraints:
- Exactly one charging option per EV.
- The number of EVs assigned to a station/time slot cannot exceed that station's configured capacity.

Capacity inequalities are converted to QUBO equality penalties using binary slack variables. This increases the number of qubits beyond the EV/station/time decision variables.

The objective combines charging duration, queue delay, detour, energy cost, amenity dissatisfaction and an illustrative battery-care preference.

## Local simulation limits

The Statevector-based QAOA implementation is capped at **20 total qubits**, including capacity slack variables. The exact classical baseline is capped at 250,000 assignment combinations. Larger scenarios are rejected with an explanatory message.

These are prototype safeguards, not fundamental QAOA limits. Larger scenarios require a different simulation/optimization architecture.

## Assumptions and limitations

- Station, queue, tariff, detour, route-position and amenity data are synthetic or user-supplied.
- Origin/destination are text labels. Route distance, average speed and station positions are user-configured approximations, not live routing.
- Charging time uses a simplified 82% effective-power factor plus a fixed connection overhead; it is not an OEM charging curve.
- The battery-care penalty is illustrative, not a battery-health model or universal charging rule.
- Queue estimates are user-configured per station/time slot; queue evolution is not simulated.
- The exact classical baseline is exact only for the configured small assignment problem.
- QAOA is heuristic. No quantum advantage is claimed.
- If no feasible bitstring appears in QAOA samples, the app identifies the exact-classical fallback.

## Project structure

```text
QuantumCharge/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── assets/
    └── quantumcharge_hero.png
```


## 2.2 quick-entry workflow

The Journey Configuration page is intentionally a single guided form. Enter origin, destination, route distance, average speed, time slots, EV count, and each EV's battery/SOC values. Choose the number of synthetic stations. On submit, the app generates station positions, capacities, charger types, queue estimates, tariffs and amenities automatically. There are no station or EV data-editor tables, avoiding accidental blank rows and missing numeric values.


## 2.3 numeric fleet and time-slot inputs

The Journey Configuration page uses integer number inputs for the number of EVs and time slots (not dropdowns). The current local Statevector QAOA implementation has a hard limit of 20 qubits. A recommended demonstration configuration is **2 EVs, 2 stations and 2 time slots**, which produces 14 qubits with the generated station capacities and is a safer, faster local demo. Larger configurations are accepted as inputs but the optimization button is deliberately disabled when the QUBO exceeds the simulator limit; this is a simulator limitation, not a UI fault.


## 2.4 one-click demo recovery

If a configured scenario exceeds the 20-qubit local Statevector limit, open **Optimize Journey** and click **Apply Recommended Demo Configuration**. This automatically resets the active demo to 2 EVs, 2 stations and 2 time slots, with station capacities 1 and 2. The resulting QUBO contains 8 assignment decision variables plus 6 capacity slack variables = **14 qubits**. Previous optimization results are cleared when the demo configuration is applied.


## 2.5 preserve user inputs when applying demo dimensions

The Apply Recommended Demo Configuration action no longer replaces the user's EV records with seeded defaults or regenerates an existing station table. It preserves the journey labels, route distance, average speed, every EV battery/SOC/fast-charge value, and the first two saved station records (including their capacities, positions, queues and tariffs). It reduces the time horizon to two slots and uses two saved stations. If the preserved EV count and station capacities still exceed 20 qubits, the app reports the actual limit rather than silently changing user data.


## 2.6 local simulator threshold

The local Statevector guardrail is now **22 qubits** rather than 20. A 22-qubit statevector contains 4,194,304 complex amplitudes (approximately 64 MiB for the raw complex128 state array), compared with 1,048,576 amplitudes at 20 qubits. Actual peak memory and runtime are higher because Qiskit, the QUBO energy array, optimizer and simulator also require memory. The threshold is a conservative application setting, not a universal Qiskit limit.

The QUBO basis-state energy calculation has also been vectorized with NumPy so it does not evaluate millions of bitstrings one by one in Python. User-entered journey, EV and station values are not overwritten to fit the threshold. Scenarios above 22 qubits remain blocked to avoid excessive local memory/runtime.


## 2.7 explainable results page

The Results & Comparison page now includes:
- A weighted score breakdown for each selected EV assignment.
- A per-vehicle explanation of the selected station/time, charge duration, queue, energy requirement/cost and amenities.
- The four lowest-scoring standalone candidate options for each EV, clearly labelled as individual rankings rather than globally feasible fleet alternatives.
- A station/time-slot capacity check for the displayed fleet schedule.
- An explanation of what a QAOA/classical score match means and why it does not prove quantum advantage.

The breakdown uses the exact score coefficients in `build_options()`:
`0.35 * approach_min + detour_min + charge_min + 1.25 * wait_min + 0.025 * energy_cost + amenity_penalty + battery_penalty`.
