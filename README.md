# ⚡ QuantumCharge

**Journey-aware EV charging optimization using a hybrid classical–quantum workflow.**

QuantumCharge explores how QUBO + QAOA can help allocate EV charging decisions across stations and time slots while considering charging duration, estimated queues, energy cost, detour, amenities and an illustrative battery-care preference.

## Features

- Futuristic dark-blue / cyan Streamlit interface.
- Synthetic EV and charging-station scenario.
- Classical preprocessing for charging duration, queue delay, tariff, detour and amenities.
- QUBO formulation for EV assignment and station/time capacity conflicts.
- One-layer QAOA circuit implemented with Qiskit.
- SciPy COBYLA parameter optimization.
- Qiskit Aer local circuit sampling.
- Exact feasible enumeration as a classical baseline.
- Results, schedule, dataset and quantum-model pages.

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

## Setup — Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If the environment already exists, activate it and run:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

## Quantum implementation

- **Qiskit:** quantum circuit construction.
- **Qiskit Statevector:** evaluates the QAOA expected QUBO energy during parameter optimization.
- **Qiskit Aer:** shot-based sampling with `AerSimulator`.
- **QAOA:** one cost layer and one X-mixer layer (`p = 1`).
- **Classical optimizer:** SciPy COBYLA optimizes gamma and beta.
- **Classical baseline:** exact enumeration finds the best feasible assignment for this small instance.

No IBM Quantum account, API key or cloud quantum hardware is required to run the prototype.

## Model overview

The binary decision variable represents whether an EV uses a particular station at a particular time slot.

The objective combines:
- estimated charging duration;
- queue delay;
- detour time;
- electricity cost;
- amenity dissatisfaction penalty;
- illustrative battery-care penalty.

Penalty terms enforce exactly one charging option per EV and prevent two EVs from occupying the same station/time slot in this capacity-one scenario.

## Important limitations

- All EV, station, queue, tariff, detour and amenity inputs are synthetic.
- Charging duration uses a simplified effective-power factor and is not an OEM charging curve.
- Battery-care penalty is illustrative. There is no universal rule that every fifth charge must be AC.
- This version does not query live charging networks, traffic, maps or station availability.
- The current QUBO is a small proof-of-concept with 12 binary decision variables.
- QAOA is heuristic. The exact classical solution is the optimum for this small instance.
- No quantum advantage is claimed.
- If QAOA samples contain no feasible bitstring, the app explicitly labels the exact-classical fallback.

## Run

```powershell
streamlit run app.py
```

Then choose **Optimize Journey** in the sidebar, run the experiment, and inspect **Results & Comparison** and **Charging Schedule**.
