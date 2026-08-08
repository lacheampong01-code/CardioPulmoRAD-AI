import torch
from src.models import CardioPulmoRAD

def run_framework_demo():
    print("--- CardioPulmoRAD-AI: Framework Simulation Demo ---")
    
    # 1. Initialize Model Architecture
    model = CardioPulmoRAD(num_classes=1)
    model.eval()
    
    # 2. Generate Synthetic Patient Inputs (1 Patient)
    dummy_chest_xray = torch.randn(1, 3, 224, 224) 
    dummy_vitals = torch.randn(1, 13) 
    
    # 3. Execute Multimodal Forward Pass
    with torch.no_grad():
        risk_probability = model(dummy_chest_xray, dummy_vitals).item()
    
    risk_percentage = risk_probability * 100
    print(f"\nCalculated Risk Probability: {risk_percentage:.2f}%")
    
    # 4. Triage & Specialist Tele-Consultation Decision Logic
    if risk_percentage < 30.0:
        tier = "LOW RISK - Manage locally at Primary Health Center."
    elif 30.0 <= risk_percentage < 70.0:
        tier = "MODERATE RISK - Initiate Tele-Consultation with Secondary Specialist."
    else:
        tier = "HIGH RISK - Emergency Dispatch to Tertiary Center. Package XAI diagnostic bundle for Off-Site Specialist."
        
    print(f"Assigned Triage Tier: {tier}\n")

if __name__ == "__main__":
    run_framework_demo()
