import subprocess, sys, pathlib

FILES = [
    "01_DPO_Loss_Core.py",
    "02_Implicit_Reward_Evaluator.py",
    "03_Length_Bias_and_SimPO.py",
    "04_Activation_Steering_and_Audit.py",
]

here = pathlib.Path(__file__).parent
for f in FILES:
    print(f"\n{'='*70}\n>>> {f}\n{'='*70}")
    result = subprocess.run([sys.executable, str(here / f)])
    if result.returncode != 0:
        print(f"[FAIL] {f} exited with {result.returncode}")
        sys.exit(result.returncode)
        
print("\nAll deliverables completed successfully.")