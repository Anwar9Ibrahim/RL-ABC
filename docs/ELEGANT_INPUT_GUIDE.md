# Elegant Input Files Guide

This guide explains the structure and requirements for the input files used by the ACC Elegant RL Training framework. The framework relies on two primary files to run simulations: the lattice file (`.lte`) and the command file (`.ele`).

The framework includes custom parsers in `rl_framework/Utils.py` and `rl_framework/Environment.py` that read, modify, and rewrite these files dynamically during the Reinforcement Learning process.

---

## 1. The Lattice File (`.lte`)

The lattice file defines the physical layout of the beamline, including all magnets, drifts, apertures, and the sequence in which they appear. 

### Key Components

#### Comments
Comments in `.lte` files start with an exclamation mark `!`.
```elegant
! This is a comment
```

#### Element Definitions
Elements like Quadrupoles (`QUAD`), Bending Magnets (`SBEND`), and Drifts (`DRIFT`) are defined with their specific parameters.
```elegant
! Format: NAME: TYPE, PARAM1=value, PARAM2=value
Q1L0:  QUAD, L=0.18, K1=0.0932, HKICK=-0.0001, VKICK=0.0001
BM1:   SBEND, L=0.87808, ANGLE=+0.785398, FSE=0.0001
LM01L0: DRIFT, L=1.0
```
*Note: The RL agent will modify specific parameters (like `K1`, `HKICK`, `VKICK`, `FSE`) based on the action space defined in your environment.*

#### Watch Points
`WATCH` elements are crucial for the RL framework. They output the beam properties at specific locations to `.sdds` files, which the environment parses to calculate rewards and observations.
```elegant
WQ1L0_1: WATCH, filename="results_120/WQ1L0_1.sdds", mode=coord
final_WP: WATCH, filename="results_120/final_WP.sdds", mode=coord
```
*Important: The framework dynamically updates the `filename` paths (e.g., changing `results_120` to the current worker's directory like `results_0`) to prevent parallel workers from overwriting each other's data.*

#### Beamline Sequences (LINE)
The `LINE` command groups elements into sequences. The final beamline must be defined as a `LINE`.
```elegant
PInject: LINE=(LM01L0, WQ1L0_1, Q1L0, LL0L1)
machine: LINE=(q, MA2, PInject, final_WP)
```

---

## 2. The Command File (`.ele`)

The command file tells Elegant *how* to run the simulation, what beam parameters to use, and what output files to generate.

### Key Components

#### Run Setup (`&run_setup`)
Defines the central momentum, the lattice file to use, and the output files for sigmas, centroids, and magnets.
```elegant
&run_setup
    lattice = elegant_input.lte
    use_beamline = machine
    p_central_mev = 400.0
    sigma = results/%s.sig
    centroid = results/%s.cen
    magnets = results/beamline.mag
    default_order=2
&end
```
*Note: The framework's parser dynamically updates the `lattice` field to point to the temporary modified `.lte` file for each episode.*

#### Twiss Output (`&twiss_output`)
Generates the Twiss parameters (beta functions, alpha, etc.).
```elegant
&twiss_output
    filename = results/twiss.twi
    matched = 0
    beta_x  =  2.0
    beta_y  =  2.0
&end
```

#### Bunched Beam (`&bunched_beam`)
Defines the initial particle distribution. The RL framework often modifies the `n_particles_per_bunch` dynamically based on the configuration.
```elegant
&bunched_beam
    n_particles_per_bunch = 1000
    emit_nx = 2000.0e-6
    emit_ny = 2000.0e-6
    beta_x  = 2.0
    beta_y  = 2.0
    sigma_s  = 5.0e-3
    sigma_dp = 4e-2
    distribution_type[0] = 3*"gaussian",
    distribution_cutoff[0] = 3*3,
&end
```

#### Track Command (`&track`)
Initiates the tracking simulation.
```elegant
&track 
&end
```

---

## 3. How the Framework Interacts with These Files

When training starts, the framework does not overwrite your original files. Instead, it uses the parsers in `Utils.py` to:

1. **Read** the base `track.ele` and `machine.lte` (or `elegant_input.lte`).
2. **Modify** the `.lte` file with the new magnet strengths chosen by the RL agent.
3. **Update Paths** in both files so that outputs are saved to a worker-specific directory (e.g., `results_0/`, `results_1/`).
4. **Write** temporary files (e.g., `updated_machine_0.lte`, `track_0.ele`).
5. **Execute** Elegant using these temporary files.
6. **Parse** the resulting `.sdds`, `.sig`, and `.cen` files to calculate the reward and next state.

### Best Practices for Users
- Always keep a backup of your original `.lte` and `.ele` files.
- Ensure that `WATCH` points are correctly placed where you need to observe the beam.
- Do not hardcode absolute paths in the `.ele` or `.lte` files; use relative paths like `results/filename.sdds`, as the parser expects this format to replace it with worker-specific directories.
