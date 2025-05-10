#!/usr/bin/env python3
"""
Test script to check if the model is responsive to different inputs.
This helps diagnose if the model is capable of producing different outputs.
"""

import argparse
import os
import torch
import numpy as np
from live_inference import LiveGameStatePredictor

def check_for_nans(predictor):
    """Check for NaN issues in the model and normalization statistics"""
    print("\n=== Checking for NaN Issues ===")
    
    # Check normalization statistics
    stats_check = predictor.check_normalization_stats()
    
    if stats_check["issues"]:
        print("Issues found in normalization statistics:")
        for issue in stats_check["issues"]:
            print(f"  - {issue}")
    else:
        print("✅ No issues found in normalization statistics")
    
    # Create test input with random values
    device = next(predictor.model.parameters()).device
    test_batch_size = 1
    test_seq_len = 10
    
    # Test with different input types
    test_inputs = [
        ("zeros", torch.zeros((test_batch_size, test_seq_len, predictor.continuous_dim), device=device)),
        ("ones", torch.ones((test_batch_size, test_seq_len, predictor.continuous_dim), device=device)),
        ("random", torch.rand((test_batch_size, test_seq_len, predictor.continuous_dim), device=device)),
        ("normal", torch.randn((test_batch_size, test_seq_len, predictor.continuous_dim), device=device))
    ]
    
    test_enum = {
        name: torch.zeros((test_batch_size, test_seq_len), dtype=torch.long, device=device)
        for name in ['stage', 'p1_action', 'p1_character', 'p2_action', 'p2_character']
    }
    
    # Target inputs
    tgt_cont = torch.zeros((test_batch_size, 5, predictor.continuous_dim), device=device)
    tgt_enum = {
        name: torch.zeros((test_batch_size, 5), dtype=torch.long, device=device)
        for name in test_enum.keys()
    }
    
    # Test each input type
    for name, test_cont in test_inputs:
        print(f"\nTesting with {name} input:")
        
        # Check for NaNs in input
        if torch.isnan(test_cont).any():
            print(f"WARNING: NaN values in {name} input")
            test_cont = torch.nan_to_num(test_cont, nan=0.0)
        
        # Run prediction
        try:
            with torch.no_grad():
                cont_pred, enum_pred = predictor.model(test_cont, test_enum, tgt_cont, tgt_enum)
            
            # Check for NaNs in output
            if torch.isnan(cont_pred).any():
                nan_count = torch.isnan(cont_pred).sum().item()
                total = cont_pred.numel()
                print(f"❌ NaN values in output: {nan_count}/{total} ({nan_count/total*100:.2f}%)")
                
                # Find where NaNs are occurring
                nan_indices = torch.where(torch.isnan(cont_pred))
                print(f"First 5 NaN positions: {list(zip(*[idx.tolist() for idx in nan_indices]))[:5]}")
            else:
                print(f"✅ No NaN values in output")
                
            # Check output statistics
            output_mean = cont_pred.mean().item()
            output_std = cont_pred.std().item()
            output_min = cont_pred.min().item()
            output_max = cont_pred.max().item()
            
            print(f"Output stats - Mean: {output_mean:.4f}, Std: {output_std:.4f}, Min: {output_min:.4f}, Max: {output_max:.4f}")
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            import traceback
            traceback.print_exc()
    
    return stats_check["issues"] == []

def main():
    parser = argparse.ArgumentParser(description="Test model responsiveness")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--stats-dir", type=str, default="./model_params", 
                        help="Directory containing normalization statistics")
    parser.add_argument("--window-size", type=int, default=10, 
                        help="Window size for context frames")
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return 1
    
    # Check if stats directory exists
    if not os.path.exists(args.stats_dir):
        print(f"Error: Stats directory not found at {args.stats_dir}")
        return 1
    
    print(f"Testing model responsiveness for: {args.model}")
    print(f"Using stats from: {args.stats_dir}")
    
    try:
        # Initialize the predictor
        predictor = LiveGameStatePredictor(
            model_path=args.model,
            window_size=args.window_size,
            stats_dir=args.stats_dir
        )
        
        # Check for NaN issues
        print("\nChecking for NaN issues...")
        nan_check_passed = check_for_nans(predictor)
        
        if not nan_check_passed:
            print("\n⚠️ WARNING: NaN issues detected. This may cause prediction problems.")
        
        # Run the responsiveness test
        cont_diff, enum_diff = predictor.test_model_responsiveness()
        
        print("\n=== Test Results ===")
        print(f"Continuous prediction difference: {cont_diff:.6f}")
        print(f"Enum prediction difference: {enum_diff:.6f}")
        
        if cont_diff < 1e-6 and enum_diff < 1e-6:
            print("\n❌ FAILED: Model produces nearly identical outputs for different inputs!")
            print("This suggests the model may not be responsive to input changes.")
            print("Possible issues:")
            print("1. The model was not trained properly")
            print("2. The model architecture has issues")
            print("3. The model weights are corrupted")
            return 1
        else:
            print("\n✅ PASSED: Model produces different outputs for different inputs as expected.")
            
            # Additional tests with varying input magnitudes
            print("\nRunning additional tests with varying input magnitudes...")
            
            device = next(predictor.model.parameters()).device
            test_batch_size = 1
            test_seq_len = 10
            
            # Create a range of inputs with increasing magnitudes
            results = []
            for magnitude in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]:
                # Create input with specific magnitude
                test_cont = torch.ones((test_batch_size, test_seq_len, predictor.continuous_dim), device=device) * magnitude
                test_enum = {
                    name: torch.zeros((test_batch_size, test_seq_len), dtype=torch.long, device=device)
                    for name in ['stage', 'p1_action', 'p1_character', 'p2_action', 'p2_character']
                }
                
                # Target inputs
                tgt_cont = torch.zeros((test_batch_size, 5, predictor.continuous_dim), device=device)
                tgt_enum = {
                    name: torch.zeros((test_batch_size, 5), dtype=torch.long, device=device)
                    for name in test_enum.keys()
                }
                
                # Run prediction
                with torch.no_grad():
                    cont_pred, _ = predictor.model(test_cont, test_enum, tgt_cont, tgt_enum)
                
                # Store results
                output_mean = cont_pred.mean().item()
                output_std = cont_pred.std().item()
                results.append((magnitude, output_mean, output_std))
            
            # Print results
            print("\nInput magnitude vs. Output statistics:")
            print("Magnitude | Output Mean | Output Std")
            print("-" * 40)
            for magnitude, mean, std in results:
                print(f"{magnitude:9.1f} | {mean:11.4f} | {std:10.4f}")
            
            # Check if outputs vary with input magnitude
            means = [r[1] for r in results]
            if max(means) - min(means) < 1e-3:
                print("\n⚠️ WARNING: Output means don't vary much with input magnitude")
                print("The model may not be sensitive to input scale")
            
            return 0
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
