# Iterative Linear-Quadratic Regulator (iLQR)

A compact implementation of the Iterative Linear-Quadratic Regulator (iLQR) algorithm for trajectory optimization and control. This repository contains code, examples, and utilities to design open-loop trajectories and locally optimal feedback controllers for nonlinear discrete-time systems.

## Quick start
1. Clone the repository and open the project directory:
```bash
git clone https://github.com/MohamedAbou-Taleb/Iterative-Linear-Quadratic-Regulator.git
cd Iterative-Linear-Quadratic-Regulator
```
2. Setup a virtual environment named `myvenv` and install the package in editable mode:

- On macOS / Linux:
```bash
python3 -m venv myvenv
source myvenv/bin/activate
pip install -e .
```

- On Windows (PowerShell):
```powershell
python -m venv myvenv
.\myvenv\Scripts\Activate.ps1
pip install -e .
```

- On Windows (CMD):
```cmd
python -m venv myvenv
myvenv\Scripts\activate.bat
pip install -e .
```


## Algorithm overview
1. Initialize control sequence and rollout to get nominal trajectory.
2. Linearize dynamics and quadratize costs along trajectory.
3. Solve Riccati-like backward pass to compute local control laws (feedforward + feedback).
4. Line-search and forward rollout with updated controls.
5. Iterate until convergence or max iterations.
