# Physics-Informed Periodic Neural Networks for Excitation Trajectory Optimization in Dynamic Identification

![PIPNN Trajectory](PIPNN_traj.gif)

This repository contains the official implementation and experimental code for the paper **Physics-Informed Periodic Neural Networks for Excitation Trajectory Optimization in Dynamic Identification**, which proposes a PIPNN-based approach for excitation trajectory design in robotic manipulator dynamics identification.

------

## 1. Environment and Dependencies

The codebase is implemented in **Python**. In general, you can install missing dependencies on demand — if a package is missing, simply install it via `pip`. The dependencies mainly consist of standard scientific Python packages and **should not cause version conflicts** under a typical Python environment.

Python >= 3.9 is recommended.

Recommended setup:

```bash
pip install -r requirements.txt
```

If you encounter a `ModuleNotFoundError`, install the required package directly:

```bash
pip install <missing-package>
```

------

## 2. Installation (Editable Mode Recommended)

It is **strongly recommended** to install this project in *editable mode* so that local code modifications are immediately reflected without reinstallation:

```bash
pip install -e .
```

This is particularly useful when modifying the PIPNN model, excitation trajectory generation, or dynamics-related utilities.

------

## 3. `pinn_excitation_traj` (Core Package)

The folder:

```text
pinn_excitation_traj/
```

contains the **PIPNN (Physics-Informed Periodic Neural Network)** implementation. This is the **core package** of the repository and provides:

- PIPNN model definition
- Physics-informed constraints
- Excitation trajectory generation framework
- Training and optimization utilities

All main methods proposed in the paper are implemented in this package.

------

## 4. `example/train.ipynb` (Example Script)

The notebook:

```text
example/train.ipynb
```

serves as a **minimal working example** demonstrating how to:

- Configure the PIPNN model
- Train an excitation trajectory
- Run the full pipeline end-to-end

This notebook is the **recommended entry point** for first-time users.

------

## 5. `comparison` (Baseline and State-of-the-Art Methods)

The folder:

```text
comparison/
```

contains implementations of the **baseline methods** and **state-of-the-art comparison methods** used in the paper, including:

- Optimization-based excitation trajectory methods
- **An Analytical Approach for Dealing with Explicit Physical Constraints in Excitation Optimization Problems of Dynamic Identification** — a **state-of-the-art method published in IEEE Transactions on Robotics (TRO)**, with official code available at:
  https://github.com/HuangShifengHUST/AnaAp.PhyConst.ExcitOptimi.DynIDen/tree/master

These implementations are provided to ensure **reproducibility and fair comparison** with the proposed PIPNN-based approach.

------

## 6. `dynamic` (Dynamics and Regression Matrix Utilities)

The folder:

```text
dynamic/
```

provides auxiliary tools for **generating the regressor matrix** used in robot dynamics identification.

⚠️ **Important Note**:

- If you want to apply this framework to a **different robotic manipulator**, you must:
  1. Modify the corresponding **kinematic and dynamic parameters** (e.g., DH/MDH parameters, link properties).
  2. Regenerate the regression matrix using the updated model.

This design allows the framework to be **easily extended to other robots**, provided the correct dynamics are supplied.

------

## 7. Citation

If you find this work useful, please cite:

> Physics-Informed Periodic Neural Networks for Excitation Trajectory Optimization in Dynamic Identification

BibTeX information will be provided after publication.

------

## 8. License

This project is released for **research and academic use only**.

- **Commercial use is strictly prohibited** without explicit permission.
- Any commercial usage, redistribution, or integration into proprietary systems **requires prior written consent from the authors of this paper**.
