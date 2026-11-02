# PS04: DPO, SimPO, and Activation Steering (SEC-04-ORI)

**Course:** Conversational AI - Winter Semester 2026/27  
**Mission:** Week 4 - Alignment Pipelines, Preference Optimization & Steering Core  
**Starship Operational Unit:** USSE Noosphere Bridge Crew  
**Squadron:** Chomsky

## Mission Overview

This mission implements preference optimization (DPO and SimPO) and inference-time activation steering from first principles in PyTorch, without high-level training wrappers such as Hugging Face TRL.

## Deliverables

### Part 1: DPO Loss Core (`01_DPO_Loss_Core.py`)

- Implements `get_batch_logps()` with `torch.gather` and padding masks.
- Implements the numerically stable `DPOLoss(nn.Module)`.
- Checks that backpropagation updates the policy while the reference model remains frozen.

### Part 2: Implicit Reward Evaluator (`02_Implicit_Reward_Evaluator.py`)

- Computes implicit rewards:  
  `r_hat(x, y) = beta * (log pi_policy(y|x) - log pi_ref(y|x))`.
- Reports pairwise accuracy and Bradley-Terry choice probabilities.
- Generates `bt_calibration.png`, a reliability diagram comparing predicted probabilities with preference labels.

### Part 3: Length Bias and SimPO (`03_Length_Bias_and_SimPO.py`)

- Measures the Pearson correlation between response-length differences and preference margins.
- Implements the reference-free `SimP0Loss(nn.Module)` using average per-token log probabilities (`gamma = 0.5`, `beta = 2.0`). The `0` in `SimP0Loss` is a zero, matching the class name in the script.
- Compares memory use for DPO (policy and reference models) and SimPO (policy only). Without CUDA, the script reports a CPU model-size proxy instead of GPU VRAM.

### Part 4: Activation Steering and Audit (`04_Activation_Steering_and_Audit.py`)

- Extracts a truthfulness-versus-sycophancy contrast vector using Linear Artificial Tomography (LAT).
- Implements `ActivationSteeringHook`, which adds `alpha * v_steer` to a decoder layer's output during inference.
- Audits a sycophancy-marker-based truthfulness proxy and arithmetic reasoning accuracy over steering multipliers. This is a small demonstration audit, not a general safety evaluation.
- Uses `Qwen/Qwen2.5-0.5B-Instruct` by default. Set `PS04_MODEL` to use a different compatible Hugging Face model.

## Setup

Python 3.10 or newer is recommended. Run these commands from the `mission-4` directory.

### Windows PowerShell

The `py` launcher avoids issues where `python` resolves to the Microsoft Store execution alias.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install transformers accelerate matplotlib numpy scipy
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python --version
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install transformers accelerate matplotlib numpy scipy
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install transformers accelerate matplotlib numpy scipy
```

The PyTorch command above installs the default build for your platform. For NVIDIA GPU support, install a CUDA-enabled build compatible with your Python version, operating system, and driver using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). CUDA wheel availability depends on the Python version; for example, the CUDA 12.1 index may not provide wheels for every Python release. Verify GPU support with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Run and Verify

With the virtual environment activated and the current directory set to `mission-4`, run individual parts:

```bash
python 01_DPO_Loss_Core.py
python 02_Implicit_Reward_Evaluator.py
python 03_Length_Bias_and_SimPO.py
python 04_Activation_Steering_and_Audit.py
```

Or run the convenience script to execute the four components in sequence. It stops if a component fails:

```bash
python 00_run_all.py
```

### Expected behavior

- **Part 1:** Prints successful checks for `get_batch_logps` and DPO gradient isolation.
- **Part 2:** Prints pairwise accuracy and calibration bins, and saves `bt_calibration.png`. The example uses synthetic data.
- **Part 3:** Prints a Pearson correlation, SimPO loss, and memory comparison. Some values use generated data and can vary between runs. On a CPU-only PyTorch build, CUDA VRAM profiling is unavailable and the script uses a CPU model-size proxy.
- **Part 4:** Downloads the selected model the first time it runs, extracts a steering vector, and prints truthfulness-proxy and reasoning scores for alpha values `-20`, `-10`, `0`, `10`, and `20`. Scores depend on the model and hardware; CPU inference may be slow.

## Jupyter Notebook

The `ps04_solution.ipynb` notebook provides a unified view of the mission:

1. Open it in VS Code.
2. Select the mission's `.venv` as the notebook kernel.
3. Run all cells.

## Git Commit

To commit the README change, run these commands from the repository root and adjust the branch name if needed:

```powershell
git add mission-4/README.md
git commit -m "docs(ps04): fix README markdown formatting"
git push origin main
```
