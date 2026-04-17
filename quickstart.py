#!/usr/bin/env python3
"""
IRDAS QUICK START GUIDE

This file serves as the entry point for the In-race Data Augmentation System.
Follow the steps below to get started.
"""

import sys

def print_welcome():
    """Print welcome message."""
    print("\n" + "="*70)
    print("IRDAS: In-race Data Augmentation System")
    print("Welcome to IRDAS - Vehicle Model Correction & State Estimation")
    print("="*70)


def print_menu():
    """Print main menu."""
    print("\nWhat would you like to do?\n")
    print("1. Run quick demo (30 seconds, shows all features)")
    print("2. Run full training & evaluation (30 minutes)")
    print("3. Run specific test scenario")
    print("4. View documentation")
    print("5. Exit")
    print()


def run_demo():
    """Run the demo script."""
    print("\nLaunching demo...")
    print("This will show you IRDAS with Kalman filter and parameter adaptation")
    print("running on simulated vehicle data with intentional parameter mismatch.\n")
    
    try:
        from demo import main
        main()
    except Exception as e:
        print(f"Error running demo: {e}")
        import traceback
        traceback.print_exc()
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")


def run_full_training():
    """Run full training and evaluation."""
    print("\nLaunching full training & evaluation...")
    print("This will:")
    print("  1. Train neural network on residual dynamics (5-10 minutes)")
    print("  2. Run multiple test scenarios (highway, aggressive, slalom)")
    print("  3. Generate performance report\n")
    
    response = input("Continue? (y/n): ").lower()
    if response == 'y':
        try:
            from train_test import run_full_evaluation
            run_full_evaluation(device='cpu')
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure all dependencies are installed:")
            print("  pip install -r requirements.txt")


def run_test_scenario():
    """Run specific test scenario."""
    print("\nAvailable test scenarios:")
    print("1. Highway driving")
    print("2. Aggressive handling")
    print("3. Slalom maneuver")
    print()
    
    choice = input("Select scenario (1-3): ").strip()
    
    try:
        from train_test import (
            test_scenario_1_highway_driving,
            test_scenario_2_aggressive_handling,
            test_scenario_3_slalom,
            create_baseline_params
        )
        from irdas_main import IRDAS
        
        baseline_params = create_baseline_params()
        irdas = IRDAS(baseline_params, device='cpu', use_nn=False, use_rls=True)
        irdas.initialize_real_vehicle(seed=42)
        
        if choice == '1':
            test_scenario_1_highway_driving(irdas, duration_seconds=20)
        elif choice == '2':
            test_scenario_2_aggressive_handling(irdas, duration_seconds=15)
        elif choice == '3':
            test_scenario_3_slalom(irdas, duration_seconds=20)
        else:
            print("Invalid choice")
    
    except Exception as e:
        print(f"Error: {e}")


def show_documentation():
    """Show key documentation sections."""
    print("\n" + "="*70)
    print("IRDAS DOCUMENTATION")
    print("="*70)
    
    print("\n1. SYSTEM OVERVIEW")
    print("-" * 70)
    print("""
IRDAS (In-race Data Augmentation System) integrates:
  • Real vehicle simulator: Creates mismatched vehicle data
  • Kalman filter: Estimates state from noisy sensors
  • Neural network: Learns residual dynamics corrections
  • Parameter adapter: Tunes tire/aerodynamic coefficients online

The system operates in a loop:
  Real Data → Kalman Filter → State Estimate
     ↓
  Parameter Adapter updates model
     ↓
  Neural Network predicts corrections
     ↓
  Improved model for next iteration
    """)
    
    print("\n2. FILES & PURPOSE")
    print("-" * 70)
    print("""
Core Components:
  • params.py              - Vehicle parameters (mass, tires, aerodynamics)
  • twin_track.py          - Baseline vehicle dynamics model (13-state bicycle)
  
IRDAS Modules:
  • simulator.py           - Real vehicle simulator with parameter mismatch
  • kalman_filter.py       - Extended Kalman Filter for state estimation
  • residual_network.py    - Neural network for residual dynamics learning
  • parameter_adapter.py   - Online parameter adaptation (RLS algorithm)
  • irdas_main.py          - Main integrated IRDAS system
  
Utilities:
  • irdas_config.py        - Configuration templates
  • train_test.py          - Training and testing scripts
  • demo.py                - Simple demo script
  • README.md              - Full documentation
  • requirements.txt       - Python dependencies
    """)
    
    print("\n3. QUICK START")
    print("-" * 70)
    print("""
Installation:
  pip install -r requirements.txt

Basic Usage:
  python demo.py
  
Full Training:
  python train_test.py --mode full

Configuration:
  Edit irdas_config.py for tuning
    """)
    
    print("\n4. KEY CONCEPTS")
    print("-" * 70)
    print("""
Residual Dynamics:
  Instead of learning the full model, the NN learns only the difference
  between the baseline model and real vehicle: r = f_real - f_baseline
  
  Advantages: smaller NN, better generalization, interpretable

Online Learning:
  Parameters adapt continuously as new data arrives
  Uses Recursive Least Squares (RLS) for efficient updates
  
Kalman Filter:
  Fuses model predictions with noisy sensor measurements
  Provides state estimates with uncertainty quantification

Parameter Bounds:
  All adaptive parameters are constrained to physical ranges
  Prevents unrealistic values (e.g., negative friction)
    """)
    
    print("\n5. PERFORMANCE METRICS")
    print("-" * 70)
    print("""
The system tracks:
  • Model Error: ||f_real(x,u) - f_baseline(x,u)||
  • Estimation Error: ||x_true - x_estimated||
  • Parameter Changes: How much parameters deviate from baseline
  • State Trajectories: Full history for offline analysis
    """)
    
    print("\n6. CUSTOMIZATION")
    print("-" * 70)
    print("""
For your specific vehicle:
  1. Update params.py with your vehicle's specifications
  2. Adjust irdas_config.py for your driving conditions
  3. Retrain neural network on your specific dynamics
  4. Validate on real vehicle data

Kalman Filter Tuning:
  • Increase Q if you trust sensors more than the model
  • Increase R if you trust the model more than sensors
  
Parameter Adaptation:
  • Lower learning_rate for conservative adaptation
  • Higher learning_rate for aggressive adaptation
  • Adjust RLS forgetting_factor (0.9 = aggressive, 0.99 = conservative)
    """)
    
    print("\n" + "="*70)
    print("For full documentation, see README.md")
    print("="*70 + "\n")


def main():
    """Main menu loop."""
    print_welcome()
    
    while True:
        print_menu()
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            run_demo()
        elif choice == '2':
            run_full_training()
        elif choice == '3':
            run_test_scenario()
        elif choice == '4':
            show_documentation()
        elif choice == '5':
            print("\nThank you for using IRDAS!")
            print("For more information, visit the README.md file.\n")
            sys.exit(0)
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    # Check dependencies
    try:
        import numpy
        import scipy
        import torch
    except ImportError:
        print("\n" + "="*70)
        print("ERROR: Missing dependencies!")
        print("="*70)
        print("\nPlease install required packages:")
        print("  pip install -r requirements.txt\n")
        sys.exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
