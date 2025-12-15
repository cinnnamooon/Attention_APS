"""
Compare results between Newton method and Attention network method
Analyzes tau distributions, RL performance, and preference accuracy
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_metrics(log_file):
    """Load training metrics from log file"""
    metrics = {
        'tau_mean': [],
        'tau_std': [],
        'tau_min': [],
        'tau_max': [],
        'rl_return': [],
        'pref_accuracy': [],
        'rm_loss': []
    }
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if 'Tau statistics:' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'mean=' in part:
                        metrics['tau_mean'].append(float(part.split('=')[1].rstrip(',')))
                    elif 'std=' in part:
                        metrics['tau_std'].append(float(part.split('=')[1].rstrip(',')))
                    elif 'min=' in part:
                        metrics['tau_min'].append(float(part.split('=')[1].rstrip(',')))
                    elif 'max=' in part:
                        metrics['tau_max'].append(float(part.split('=')[1]))

            if 'Attention Network Metrics:' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'train_loss=' in part:
                        pass  # Could add if needed
            
            # Parse RL return
            if 'ep_rew_mean' in line:
                try:
                    value = float(line.split('|')[-1].strip())
                    metrics['rl_return'].append(value)
                except:
                    pass
    
    return metrics

def plot_tau_distributions(newton_metrics, attention_metrics, output_dir):
    """Plot tau distribution comparisons"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Tau mean over time
    ax = axes[0, 0]
    if newton_metrics['tau_mean']:
        ax.plot(newton_metrics['tau_mean'], label='Newton Method', linewidth=2)
    if attention_metrics['tau_mean']:
        ax.plot(attention_metrics['tau_mean'], label='Attention Network', linewidth=2)
    ax.set_xlabel('RM Training Epoch')
    ax.set_ylabel('Mean Tau')
    ax.set_title('Tau Mean Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Tau std over time
    ax = axes[0, 1]
    if newton_metrics['tau_std']:
        ax.plot(newton_metrics['tau_std'], label='Newton Method', linewidth=2)
    if attention_metrics['tau_std']:
        ax.plot(attention_metrics['tau_std'], label='Attention Network', linewidth=2)
    ax.set_xlabel('RM Training Epoch')
    ax.set_ylabel('Std of Tau')
    ax.set_title('Tau Standard Deviation (Higher = Better Diversity)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Tau range over time
    ax = axes[1, 0]
    if newton_metrics['tau_min'] and newton_metrics['tau_max']:
        epochs = range(len(newton_metrics['tau_min']))
        ax.fill_between(epochs, newton_metrics['tau_min'], newton_metrics['tau_max'], 
                        alpha=0.3, label='Newton Method Range')
    if attention_metrics['tau_min'] and attention_metrics['tau_max']:
        epochs = range(len(attention_metrics['tau_min']))
        ax.fill_between(epochs, attention_metrics['tau_min'], attention_metrics['tau_max'], 
                        alpha=0.3, label='Attention Network Range')
    ax.set_xlabel('RM Training Epoch')
    ax.set_ylabel('Tau Range')
    ax.set_title('Tau Min-Max Range')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Distribution histogram (last epoch)
    ax = axes[1, 1]
    methods = ['Newton', 'Attention']
    means = [
        newton_metrics['tau_mean'][-1] if newton_metrics['tau_mean'] else 0,
        attention_metrics['tau_mean'][-1] if attention_metrics['tau_mean'] else 0
    ]
    stds = [
        newton_metrics['tau_std'][-1] if newton_metrics['tau_std'] else 0,
        attention_metrics['tau_std'][-1] if attention_metrics['tau_std'] else 0
    ]
    x = np.arange(len(methods))
    width = 0.35
    ax.bar(x - width/2, means, width, label='Mean', alpha=0.8)
    ax.bar(x + width/2, stds, width, label='Std', alpha=0.8)
    ax.set_ylabel('Value')
    ax.set_title('Final Tau Statistics')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tau_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved tau comparison plot to {output_dir}/tau_comparison.png")
    plt.close()

def plot_rl_performance(newton_metrics, attention_metrics, output_dir):
    """Plot RL performance comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if newton_metrics['rl_return']:
        ax.plot(newton_metrics['rl_return'], label='Newton Method', linewidth=2, alpha=0.8)
    if attention_metrics['rl_return']:
        ax.plot(attention_metrics['rl_return'], label='Attention Network', linewidth=2, alpha=0.8)

    ax.axhline(y=3448, color='green', linestyle='--', label='Baseline (No APS)', linewidth=2)
    ax.axhline(y=3141, color='red', linestyle='--', label='Previous APS (Newton)', linewidth=2)
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Episode Return')
    ax.set_title('RL Performance Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rl_performance.png'), dpi=300, bbox_inches='tight')
    print(f"Saved RL performance plot to {output_dir}/rl_performance.png")
    plt.close()

def print_summary(newton_metrics, attention_metrics):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print("\n--- Tau Statistics (Final Epoch) ---")
    if newton_metrics['tau_mean']:
        print(f"Newton Method:")
        print(f"  Mean: {newton_metrics['tau_mean'][-1]:.4f}")
        print(f"  Std:  {newton_metrics['tau_std'][-1]:.4f}")
        print(f"  Min:  {newton_metrics['tau_min'][-1]:.4f}")
        print(f"  Max:  {newton_metrics['tau_max'][-1]:.4f}")
    
    if attention_metrics['tau_mean']:
        print(f"\nAttention Network:")
        print(f"  Mean: {attention_metrics['tau_mean'][-1]:.4f}")
        print(f"  Std:  {attention_metrics['tau_std'][-1]:.4f}")
        print(f"  Min:  {attention_metrics['tau_min'][-1]:.4f}")
        print(f"  Max:  {attention_metrics['tau_max'][-1]:.4f}")
    
    print("\n--- RL Performance (Best Return) ---")
    if newton_metrics['rl_return']:
        print(f"Newton Method: {max(newton_metrics['rl_return']):.2f}")
    if attention_metrics['rl_return']:
        print(f"Attention Network: {max(attention_metrics['rl_return']):.2f}")
    print(f"Baseline (No APS): 3448.00 (reference)")
    print(f"Previous APS: 3141.00 (reference)")
    
    print("\n--- Key Insights ---")
    if newton_metrics['tau_std'] and attention_metrics['tau_std']:
        newton_std = newton_metrics['tau_std'][-1]
        attention_std = attention_metrics['tau_std'][-1]
        if attention_std > newton_std:
            improvement = ((attention_std - newton_std) / newton_std) * 100
            print(f"✓ Attention network increases tau diversity by {improvement:.1f}%")
        else:
            print(f"✗ Attention network has lower tau diversity")
    
    if newton_metrics['rl_return'] and attention_metrics['rl_return']:
        newton_best = max(newton_metrics['rl_return'])
        attention_best = max(attention_metrics['rl_return'])
        if attention_best > newton_best:
            improvement = attention_best - newton_best
            print(f"✓ Attention network improves RL return by {improvement:.2f}")
        else:
            print(f"✗ Attention network has lower RL return")
        
        if attention_best > 3448:
            print(f"✓ Attention network beats baseline!")
        if newton_best > 3448:
            print(f"✓ Newton method beats baseline!")
    
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Compare Newton vs Attention methods')
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Directory containing comparison results (e.g., experiments/comparison_20240101_120000)')
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        print(f"Error: Directory {experiment_dir} does not exist!")
        return
    
    newton_log = experiment_dir / 'newton_method.log'
    attention_log = experiment_dir / 'attention_method.log'
    
    print(f"Loading metrics from {experiment_dir}...")
    
    newton_metrics = {'tau_mean': [], 'tau_std': [], 'tau_min': [], 'tau_max': [], 'rl_return': []}
    attention_metrics = {'tau_mean': [], 'tau_std': [], 'tau_min': [], 'tau_max': [], 'rl_return': []}
    
    if newton_log.exists():
        print(f"  Loading Newton method log...")
        newton_metrics = load_metrics(newton_log)
    else:
        print(f"  Warning: {newton_log} not found!")
    
    if attention_log.exists():
        print(f"  Loading Attention network log...")
        attention_metrics = load_metrics(attention_log)
    else:
        print(f"  Warning: {attention_log} not found!")
    
    output_dir = experiment_dir / 'analysis'
    output_dir.mkdir(exist_ok=True)

    print("\nGenerating comparison plots...")
    plot_tau_distributions(newton_metrics, attention_metrics, output_dir)
    plot_rl_performance(newton_metrics, attention_metrics, output_dir)

    print_summary(newton_metrics, attention_metrics)
    
    print(f"\nAnalysis complete! Results saved in {output_dir}")

if __name__ == '__main__':
    main()
