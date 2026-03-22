# Elegant Input Files Guide

This guide explains the structure and requirements for the input files used by the ACC Elegant RL Training framework. The framework relies on two primary files to run simulations: the lattice file (`.lte`) and the command file (`.ele`).

The framework includes custom parsers in `rl_framework/Utils.py` and `rl_framework/Environment.py` that read, modify, and rewrite these files dynamically during the Reinforcement Learning process.

---

## 1. The Lattice File (`.lte`)

The lattice file defines the physical layout of the beamline, including all magnets, drifts, apertures, and the sequence in which they appear. You can use `machine.lte` as a reference template.

### Key Focus for Users: Element Definitions
As a user, **your primary responsibility is to define the elements correctly using standard Elegant keywords** (`QUAD`, `SBEND`, `DRIFT`, `MAXAMP`, etc.). 

```elegant
! Format: NAME: TYPE, PARAM1=value, PARAM2=value
Q1L0:  QUAD, L=0.18, K1=0.0932, HKICK=-0.0001, VKICK=0.0001
BM1:   SBEND, L=0.87808, ANGLE=+0.785398, FSE=0.0001
LM01L0: DRIFT, L=1.0
```
*Note: The RL agent will dynamically modify specific parameters (like `K1`, `HKICK`, `VKICK`, `FSE`) based on the action space defined in your environment.*

### Beamline Sequences (LINE)
You must group your elements into sequences using the `LINE` command. The final beamline must be defined as a `LINE`.
```elegant
PInject: LINE=(LM01L0, Q1L0, LL0L1)
machine: LINE=(q, MA2, PInject)
```

### What You DON'T Need to Worry About: Watch Points & Paths
`WATCH` elements are crucial for the RL framework to observe beam properties. However, **you do not need to manually place `WATCH` points or specify their output paths in your `.lte` file.** 

The Python framework automatically:
1. Injects the necessary `WATCH` points into the beamline.
2. Handles routing the outputs to the correct worker directories (e.g., `results_0/`) to prevent parallel workers from overwriting each other's data.

---

## 2. The Command File (`.ele`)

The command file tells Elegant *how* to run the simulation, what beam parameters to use, and what output files to generate. You can use `track.ele` as a reference template.

### Key Components

#### Run Setup (`&run_setup`)
Defines the central momentum, the lattice file to use, and the output files for sigmas, centroids, and magnets. 

**Important:** Just like the `.lte` file, the Python parsers will automatically update the paths and filenames here. You can leave them as generic `results/` paths:
```elegant
&run_setup
    lattice = machine.lte
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

1. **Read** the base `track.ele` and `machine.lte`.
2. **Modify** the `.lte` file with the new magnet strengths chosen by the RL agent.
3. **Inject Watch Points** automatically into the beamline to gather observations.
4. **Update Paths** automatically in both files so that outputs are saved to a worker-specific directory (e.g., `results_0/`, `results_1/`). You do not need to worry about path collisions between parallel workers.
5. **Write** temporary files (e.g., `updated_machine_0.lte`, `track_0.ele`).
6. **Execute** Elegant using these temporary files.
7. **Parse** the resulting `.sdds`, `.sig`, and `.cen` files to calculate the reward and next state.

### Best Practices for Users
- **Use the provided examples:** The repository includes working example files (`machine.lte` and `track.ele`). Use these as references when creating your own beamline configurations.
- **Focus on the physics:** Ensure your elements (`QUAD`, `SBEND`, `DRIFT`) and beamline sequences (`LINE`) are defined correctly.
- **Keep backups:** Always keep a backup of your original `.lte` and `.ele` files.
- **Let the framework handle the rest:** Do not manually add `WATCH` points or worry about output directory paths; just use generic `results/` prefixes in your base files.
