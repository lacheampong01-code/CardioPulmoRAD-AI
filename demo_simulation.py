import time
import torch
import torch.nn as nn

# ANSI Color & Style Constants
class Style:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

def print_banner():
    print(f"\n{Style.CYAN}{Style.BOLD}" + "═" * 70)
    print("  CARD IOPULMORAD-AI | Multimodal Triage Architecture Simulation  ")
    print("═" * 70 + f"{Style.RESET}\n")

def print_step(step_num, title):
    print(f"{Style.BLUE}{Style.BOLD}[Step {step_num}]{Style.RESET} {Style.BOLD}{title}{Style.RESET}")

def run_simulation():
    print_banner()

    # Step 1: Framework Setup
    print_step(1, "Initializing Network Architecture")
    time.sleep(0.4)
    print(f"  ├─ {Style.DIM}Vision Branch:{Style.RESET}   DenseNet-121 Backbone  →  Vector V_img  ∈ ℝ¹⁰²⁴")
    print(f"  ├─ {Style.DIM}Clinical Branch:{Style.RESET} Tabular MLP (13 Vitals) →  Vector V_clin ∈ ℝ¹²⁸")
    print(f"  └─ {Style.DIM}Fusion Method:{Style.RESET}   Intermediate Fusion      →  Vector V_fused ∈ ℝ¹¹⁵²")
    print(f"  {Style.GREEN}✔ Architecture compiled successfully.{Style.RESET}\n")

    # Step 2: Input Generation
    print_step(2, "Generating Simulated Multimodal Input Tensors")
    time.sleep(0.4)
    img_tensor = torch.randn(1, 3, 224, 224)
    clin_tensor = torch.randn(1, 13)
    print(f"  ├─ Radiograph Input Tensor Shape: {Style.YELLOW}{list(img_tensor.shape)}{Style.RESET}")
    print(f"  └─ Clinical Vitals Vector Shape:  {Style.YELLOW}{list(clin_tensor.shape)}{Style.RESET}")
    print(f"  {Style.GREEN}✔ Input tensors staged in memory.{Style.RESET}\n")

    # Step 3: Forward Pass Simulation
    print_step(3, "Executing Forward Pass & Feature Concatenation")
    time.sleep(0.5)
    fused_tensor = torch.cat((torch.randn(1, 1024), torch.randn(1, 128)), dim=1)
    
    # Simulated prediction logit/prob
    simulated_prob = 0.8432  # 84.32%
    
    print(f"  ├─ Concatenated Vector V_fused:   {Style.YELLOW}{list(fused_tensor.shape)}{Style.RESET}")
    print(f"  └─ Decompensation Probability:     {Style.BOLD}{Style.RED}{simulated_prob * 100:.2f}%{Style.RESET}\n")

    # Step 4: Triage Classification Card
    print_step(4, "Evaluating Triage Decision Protocol")
    time.sleep(0.3)

    if simulated_prob >= 0.70:
        status_color = Style.RED
        status_text = "HIGH RISK (Emergency Transfer Recommended)"
        action_text = "Trigger Tele-Triage Gateway & Transmit Grad-CAM + SHAP Bundle"
    elif simulated_prob >= 0.30:
        status_color = Style.YELLOW
        status_text = "MODERATE RISK (Async Consultation Needed)"
        action_text = "Queue for Specialist Review within 2 Hours"
    else:
        status_color = Style.GREEN
        status_text = "LOW RISK (Local PHC Management)"
        action_text = "Routine Follow-up & Standard Protocol"

    print(f"\n┌" + "─" * 68 + "┐")
    print(f"│ {Style.BOLD}DECISION REPORT{Style.RESET}" + " " * 53 + "│")
    print(f"├" + "─" * 68 + "┤")
    print(f"│  {Style.BOLD}Risk Category:{Style.RESET}   {status_color}{Style.BOLD}{status_text}{Style.RESET}")
    print(f"│  {Style.BOLD}Action Protocol:{Style.RESET} {action_text}")
    print(f"└" + "─" * 68 + "┘\n")

    print(f"{Style.GREEN}{Style.BOLD}[✔] Simulation completed flawlessly.{Style.RESET}\n")

if __name__ == "__main__":
    run_simulation()
