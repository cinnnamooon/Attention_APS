"""
Automated comparison script: Run both APS methods and generate comparison plots
This script will:
1. Run original Newton-based APS
2. Run new Attention-based APS
3. Parse training logs
4. Generate comparison plots
5. Print summary statistics
"""

import os
import sys
import subprocess
import time
import json
import re
from datetime import datetime
from pathlib import Path
import argparse

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not available. Will generate data files but no plots.")
    PLOTTING_AVAILABLE = False


class ExperimentRunner:
    def __init__(self, env_id='AntBulletEnv-v0', seed=42, n_rm_epochs=5,
                 tau_max=1.0, rho=0.1, rl_steps=1000000, output_dir=None):
        self.env_id = env_id
        self.seed = seed
        self.n_rm_epochs = n_rm_epochs
        self.tau_max = tau_max
        self.rho = rho
        self.rl_steps = rl_steps
        
        # Create output directory (or reuse if provided)
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"experiments/auto_comparison_{timestamp}")
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"AUTOMATED APS COMPARISON EXPERIMENT")
        print(f"{'='*70}")
        print(f"Output directory: {self.output_dir}")
        print(f"Environment: {env_id}")
        print(f"Seed: {seed}")
        print(f"RM Epochs: {n_rm_epochs}")
        print(f"Tau Max: {tau_max}")
        print(f"Rho: {rho}")
        print(f"RL Steps: {rl_steps}")
        print(f"{'='*70}\n")
    
    def run_experiment(self, method_name, use_attention=False):
        """Run a single experiment"""
        print(f"\n{'='*70}")
        print(f"Running {method_name}...")
        print(f"{'='*70}")
        
        # Build command
        cmd = [
            sys.executable,  # Use current Python interpreter
            "train_rlhf.py",
            "--env_id", self.env_id,
            "--seed", str(self.seed),
            "--enable_individualized_tau",
            "--n_rm_epochs", str(self.n_rm_epochs),
            "--tau_max", str(self.tau_max),
            "--rho", str(self.rho),
            "--rl_steps", str(self.rl_steps),
            "--output_path", str(self.output_dir / method_name)
        ]
        
        if use_attention:
            cmd.extend([
                "--use_attention_tau",
                "--attention_hidden_dim", "128",
                "--attention_num_heads", "4",
                "--attention_num_layers", "2",
                "--attention_lr", "0.0001",
                "--train_attention_freq", "5"
            ])
        
        # Run experiment
        log_file = self.output_dir / f"{method_name}.log"
        start_time = time.time()
        
        print(f"Command: {' '.join(cmd)}")
        print(f"Log file: {log_file}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Stream output to both console and file
                for line in process.stdout:
                    print(line, end='')
                    f.write(line)
                    f.flush()
                
                process.wait()
                
            elapsed = time.time() - start_time
            print(f"\n{method_name} completed in {elapsed/60:.1f} minutes")
            
            if process.returncode != 0:
                print(f"Warning: {method_name} exited with code {process.returncode}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error running {method_name}: {e}")
            return False
    
    def parse_log_file(self, log_file):
        """Parse training log to extract metrics.

        Supports multiple log formats:
        - Newton: single-line "Tau statistics: mean=..., std=..., min=..., max=..." + "At bounds: ..."
        - Attention: multi-line block under "Attention Network Metrics:" with "Tau mean:" etc.
        - Reward model: repeated "epoch_loss:" / "epoch_acc:" lines and optional "test acc:" / "test ece:".
        """

        metrics = {
            'tau_mean': [],
            'tau_std': [],
            'tau_min': [],
            'tau_max': [],
            'tau_at_upper': [],
            'tau_at_lower': [],
            'rl_returns': [],
            'rm_losses': [],
            'rm_acc': [],
            'rm_test_acc': [],
            'rm_test_ece': [],
            'attention_losses': []
        }

        def _parse_float_after_colon(text):
            # Handles "key: 0.123" and "key: 0.0%"
            try:
                value_text = text.split(':', 1)[1].strip().replace('%', '')
                return float(value_text)
            except Exception:
                return None

        in_attention_metrics_block = False

        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for raw_line in f:
                line = raw_line.strip()

                if not line:
                    continue

                # Reward model training metrics
                if line.startswith('epoch_loss:'):
                    value = _parse_float_after_colon(line)
                    if value is not None:
                        metrics['rm_losses'].append(value)
                    continue

                if line.startswith('epoch_acc:'):
                    value = _parse_float_after_colon(line)
                    if value is not None:
                        metrics['rm_acc'].append(value)
                    continue

                if line.startswith('test acc:'):
                    value = _parse_float_after_colon(line)
                    if value is not None:
                        metrics['rm_test_acc'].append(value)
                    continue

                if line.startswith('test ece:'):
                    value = _parse_float_after_colon(line)
                    if value is not None:
                        metrics['rm_test_ece'].append(value)
                    continue

                # Parse tau statistics (Newton-style single line)
                if 'Tau statistics:' in line:
                    try:
                        mean_match = re.search(r'mean=([\d.]+)', line)
                        std_match = re.search(r'std=([\d.]+)', line)
                        min_match = re.search(r'min=([\d.]+)', line)
                        max_match = re.search(r'max=([\d.]+)', line)

                        if mean_match:
                            metrics['tau_mean'].append(float(mean_match.group(1)))
                        if std_match:
                            metrics['tau_std'].append(float(std_match.group(1)))
                        if min_match:
                            metrics['tau_min'].append(float(min_match.group(1)))
                        if max_match:
                            metrics['tau_max'].append(float(max_match.group(1)))
                    except Exception:
                        pass
                    continue

                # Parse boundary percentages (Newton-style single line)
                if 'At bounds:' in line:
                    try:
                        lower_match = re.search(r'lower=([\d.]+)%', line)
                        upper_match = re.search(r'upper=([\d.]+)%', line)

                        if lower_match:
                            metrics['tau_at_lower'].append(float(lower_match.group(1)))
                        if upper_match:
                            metrics['tau_at_upper'].append(float(upper_match.group(1)))
                    except Exception:
                        pass
                    continue

                # Parse RL returns (from evaluation)
                if 'ep_rew_mean' in line:
                    try:
                        # Format: | ep_rew_mean        | 1234.56  |
                        parts = line.split('|')
                        if len(parts) >= 3:
                            value = float(parts[2].strip())
                            metrics['rl_returns'].append(value)
                    except Exception:
                        pass
                    continue

                # Attention network metrics block
                if line.startswith('Attention Network Metrics:'):
                    in_attention_metrics_block = True
                    continue

                if in_attention_metrics_block:
                    if line.startswith('--- End Attention Network Training'):
                        in_attention_metrics_block = False
                        continue

                    if line.startswith('Loss:'):
                        value = _parse_float_after_colon(line)
                        if value is not None:
                            metrics['attention_losses'].append(value)
                        continue

                    if line.startswith('Tau mean:'):
                        value = _parse_float_after_colon(line)
                        if value is not None:
                            metrics['tau_mean'].append(value)
                        continue

                    if line.startswith('Tau std:'):
                        value = _parse_float_after_colon(line)
                        if value is not None:
                            metrics['tau_std'].append(value)
                        continue

                    if line.startswith('Tau min:'):
                        value = _parse_float_after_colon(line)
                        if value is not None:
                            metrics['tau_min'].append(value)
                        continue

                    if line.startswith('Tau max:'):
                        value = _parse_float_after_colon(line)
                        if value is not None:
                            metrics['tau_max'].append(value)
                        continue

                    if line.startswith('Tau at upper:'):
                        value = _parse_float_after_colon(line)
                        if value is not None:
                            metrics['tau_at_upper'].append(value)
                        continue

                    if line.startswith('Tau at lower:'):
                        value = _parse_float_after_colon(line)
                        if value is not None:
                            metrics['tau_at_lower'].append(value)
                        continue

        return metrics
    
    def generate_comparison_plots(self, newton_metrics, attention_metrics):
        """Generate comparison plots"""
        if not PLOTTING_AVAILABLE:
            print("Matplotlib not available. Skipping plot generation.")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Tau mean over time
        ax1 = plt.subplot(3, 3, 1)
        if newton_metrics['tau_mean']:
            ax1.plot(newton_metrics['tau_mean'], label='Newton', linewidth=2, marker='o', markersize=4)
        if attention_metrics['tau_mean']:
            ax1.plot(attention_metrics['tau_mean'], label='Attention', linewidth=2, marker='s', markersize=4)
        ax1.set_xlabel('Update')
        ax1.set_ylabel('Mean Tau')
        ax1.set_title('Tau Mean Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Tau std over time
        ax2 = plt.subplot(3, 3, 2)
        if newton_metrics['tau_std']:
            ax2.plot(newton_metrics['tau_std'], label='Newton', linewidth=2, marker='o', markersize=4)
        if attention_metrics['tau_std']:
            ax2.plot(attention_metrics['tau_std'], label='Attention', linewidth=2, marker='s', markersize=4)
        ax2.set_xlabel('Update')
        ax2.set_ylabel('Std of Tau')
        ax2.set_title('Tau Diversity (Higher = Better)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Percentage at upper bound
        ax3 = plt.subplot(3, 3, 3)
        if newton_metrics['tau_at_upper']:
            ax3.plot(newton_metrics['tau_at_upper'], label='Newton', linewidth=2, marker='o', markersize=4)
        if attention_metrics['tau_at_upper']:
            ax3.plot(attention_metrics['tau_at_upper'], label='Attention', linewidth=2, marker='s', markersize=4)
        ax3.set_xlabel('Update')
        ax3.set_ylabel('% at Upper Bound')
        ax3.set_title('Tau Clustering at Upper Bound (Lower = Better)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Tau range (min-max)
        ax4 = plt.subplot(3, 3, 4)
        if newton_metrics['tau_min'] and newton_metrics['tau_max']:
            epochs = range(len(newton_metrics['tau_min']))
            ax4.fill_between(epochs, newton_metrics['tau_min'], newton_metrics['tau_max'],
                            alpha=0.3, label='Newton Range')
            if newton_metrics['tau_mean']:
                ax4.plot(newton_metrics['tau_mean'][:len(epochs)], 'b-', linewidth=2, label='Newton Mean')
        elif newton_metrics['tau_mean'] and newton_metrics['tau_std']:
            epochs = range(min(len(newton_metrics['tau_mean']), len(newton_metrics['tau_std'])))
            mean = np.array(newton_metrics['tau_mean'][:len(epochs)])
            std = np.array(newton_metrics['tau_std'][:len(epochs)])
            ax4.fill_between(epochs, mean - std, mean + std, alpha=0.2, label='Newton Mean±Std')
            ax4.plot(mean, 'b-', linewidth=2, label='Newton Mean')

        if attention_metrics['tau_min'] and attention_metrics['tau_max']:
            epochs = range(len(attention_metrics['tau_min']))
            ax4.fill_between(epochs, attention_metrics['tau_min'], attention_metrics['tau_max'],
                            alpha=0.3, label='Attention Range')
            if attention_metrics['tau_mean']:
                ax4.plot(attention_metrics['tau_mean'][:len(epochs)], 'r-', linewidth=2, label='Attention Mean')
        elif attention_metrics['tau_mean'] and attention_metrics['tau_std']:
            epochs = range(min(len(attention_metrics['tau_mean']), len(attention_metrics['tau_std'])))
            mean = np.array(attention_metrics['tau_mean'][:len(epochs)])
            std = np.array(attention_metrics['tau_std'][:len(epochs)])
            ax4.fill_between(epochs, mean - std, mean + std, alpha=0.2, label='Attention Mean±Std')
            ax4.plot(mean, 'r-', linewidth=2, label='Attention Mean')

        ax4.set_xlabel('Update')
        ax4.set_ylabel('Tau Value')
        ax4.set_title('Tau Range')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: RL returns
        ax5 = plt.subplot(3, 3, 5)
        if newton_metrics['rl_returns']:
            ax5.plot(newton_metrics['rl_returns'], label='Newton', linewidth=2, alpha=0.8)
        if attention_metrics['rl_returns']:
            ax5.plot(attention_metrics['rl_returns'], label='Attention', linewidth=2, alpha=0.8)
        ax5.set_xlabel('Evaluation Steps')
        ax5.set_ylabel('Episode Return')
        ax5.set_title('RL Performance')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Final tau distribution (bar chart)
        ax6 = plt.subplot(3, 3, 6)
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
        ax6.bar(x - width/2, means, width, label='Mean', alpha=0.8)
        ax6.bar(x + width/2, stds, width, label='Std', alpha=0.8)
        ax6.set_ylabel('Value')
        ax6.set_title('Final Tau Statistics')
        ax6.set_xticks(x)
        ax6.set_xticklabels(methods)
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Plot 7: Percentage at lower bound
        ax7 = plt.subplot(3, 3, 7)
        if newton_metrics['tau_at_lower']:
            ax7.plot(newton_metrics['tau_at_lower'], label='Newton', linewidth=2, marker='o', markersize=4)
        if attention_metrics['tau_at_lower']:
            ax7.plot(attention_metrics['tau_at_lower'], label='Attention', linewidth=2, marker='s', markersize=4)
        ax7.set_xlabel('Update')
        ax7.set_ylabel('% at Lower Bound')
        ax7.set_title('Tau Clustering at Lower Bound')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # Plot 8: Best RL return comparison
        ax8 = plt.subplot(3, 3, 8)
        methods = ['Newton\nAPS', 'Attention\nAPS']
        returns = [
            max(newton_metrics['rl_returns']) if newton_metrics['rl_returns'] else 0,
            max(attention_metrics['rl_returns']) if attention_metrics['rl_returns'] else 0
        ]
        colors = ['blue', 'red']
        bars = ax8.bar(methods, returns, color=colors, alpha=0.7)
        ax8.set_ylabel('Best Episode Return')
        ax8.set_title('Best RL Performance')
        ax8.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, returns):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 9: Reward model training accuracy
        ax9 = plt.subplot(3, 3, 9)
        if newton_metrics.get('rm_acc'):
            ax9.plot(newton_metrics['rm_acc'], label='Newton', linewidth=2, alpha=0.9)
        if attention_metrics.get('rm_acc'):
            ax9.plot(attention_metrics['rm_acc'], label='Attention', linewidth=2, alpha=0.9)
        if newton_metrics.get('rm_acc') or attention_metrics.get('rm_acc'):
            ax9.set_xlabel('RM Epoch Step')
            ax9.set_ylabel('Training Accuracy')
            ax9.set_title('Reward Model Training Accuracy')
            ax9.legend()
            ax9.grid(True, alpha=0.3)
        else:
            ax9.text(0.5, 0.5, 'No Reward Model\nAccuracy Data', 
                    ha='center', va='center', fontsize=12)
            ax9.set_xticks([])
            ax9.set_yticks([])
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / 'comparison_plots.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved comparison plots to: {plot_file}")
        plt.close()
    
    def print_summary(self, newton_metrics, attention_metrics):
        """Print summary statistics"""
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")
        
        print("\n--- Tau Statistics (Final Epoch) ---")
        if newton_metrics['tau_mean']:
            print(f"\nNewton Method:")
            print(f"  Mean:  {newton_metrics['tau_mean'][-1]:.4f}")
            print(f"  Std:   {newton_metrics['tau_std'][-1]:.4f}")
            print(f"  Min:   {newton_metrics['tau_min'][-1]:.4f}")
            print(f"  Max:   {newton_metrics['tau_max'][-1]:.4f}")
            if newton_metrics['tau_at_upper']:
                print(f"  Upper: {newton_metrics['tau_at_upper'][-1]:.1f}%")
            if newton_metrics['tau_at_lower']:
                print(f"  Lower: {newton_metrics['tau_at_lower'][-1]:.1f}%")
        
        if attention_metrics['tau_mean']:
            print(f"\nAttention Network:")
            print(f"  Mean:  {attention_metrics['tau_mean'][-1]:.4f}")
            if attention_metrics['tau_std']:
                print(f"  Std:   {attention_metrics['tau_std'][-1]:.4f}")
            if attention_metrics['tau_min']:
                print(f"  Min:   {attention_metrics['tau_min'][-1]:.4f}")
            if attention_metrics['tau_max']:
                print(f"  Max:   {attention_metrics['tau_max'][-1]:.4f}")
            if attention_metrics['tau_at_upper']:
                print(f"  Upper: {attention_metrics['tau_at_upper'][-1]:.1f}%")
            if attention_metrics['tau_at_lower']:
                print(f"  Lower: {attention_metrics['tau_at_lower'][-1]:.1f}%")
        
        print("\n--- RL Performance (Best Return) ---")
        newton_best = max(newton_metrics['rl_returns']) if newton_metrics['rl_returns'] else 0
        attention_best = max(attention_metrics['rl_returns']) if attention_metrics['rl_returns'] else 0
        
        print(f"  Newton Method:      {newton_best:.2f}")
        print(f"  Attention Network:  {attention_best:.2f}")
        
        print("\n--- Key Insights ---")
        
        # Tau diversity comparison
        if newton_metrics['tau_std'] and attention_metrics['tau_std']:
            newton_std = newton_metrics['tau_std'][-1]
            attention_std = attention_metrics['tau_std'][-1]
            if attention_std > newton_std:
                improvement = ((attention_std - newton_std) / newton_std) * 100
                print(f"  ✓ Attention increases tau diversity by {improvement:.1f}%")
            else:
                decline = ((newton_std - attention_std) / newton_std) * 100
                print(f"  ✗ Attention decreases tau diversity by {decline:.1f}%")
        
        # Upper bound clustering
        if newton_metrics['tau_at_upper'] and attention_metrics['tau_at_upper']:
            newton_upper = newton_metrics['tau_at_upper'][-1]
            attention_upper = attention_metrics['tau_at_upper'][-1]
            reduction = newton_upper - attention_upper
            print(f"  ✓ Attention reduces upper bound clustering by {reduction:.1f}%")
            print(f"    (Newton: {newton_upper:.1f}% → Attention: {attention_upper:.1f}%)")
        
        # RL performance comparison
        if attention_best > newton_best:
            improvement = attention_best - newton_best
            print(f"  ✓ Attention improves RL return by {improvement:.2f}")
        elif attention_best < newton_best:
            decline = newton_best - attention_best
            print(f"  ✗ Attention decreases RL return by {decline:.2f}")
        
        
        print(f"\n{'='*70}\n")
        
        # Save summary to file
        summary_file = self.output_dir / 'summary.txt'
        with open(summary_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("COMPARISON SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Experiment: {self.output_dir.name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Environment: {self.env_id}\n")
            f.write(f"Seed: {self.seed}\n")
            f.write(f"RM Epochs: {self.n_rm_epochs}\n")
            f.write(f"Tau Max: {self.tau_max}\n")
            f.write(f"Rho: {self.rho}\n\n")
            
            f.write("--- Tau Statistics (Final Epoch) ---\n\n")
            if newton_metrics['tau_mean']:
                f.write("Newton Method:\n")
                f.write(f"  Mean:  {newton_metrics['tau_mean'][-1]:.4f}\n")
                f.write(f"  Std:   {newton_metrics['tau_std'][-1]:.4f}\n")
                f.write(f"  Min:   {newton_metrics['tau_min'][-1]:.4f}\n")
                f.write(f"  Max:   {newton_metrics['tau_max'][-1]:.4f}\n")
                if newton_metrics['tau_at_upper']:
                    f.write(f"  Upper: {newton_metrics['tau_at_upper'][-1]:.1f}%\n")
                if newton_metrics['tau_at_lower']:
                    f.write(f"  Lower: {newton_metrics['tau_at_lower'][-1]:.1f}%\n")
            
            if attention_metrics['tau_mean']:
                f.write("\nAttention Network:\n")
                f.write(f"  Mean:  {attention_metrics['tau_mean'][-1]:.4f}\n")
                if attention_metrics['tau_std']:
                    f.write(f"  Std:   {attention_metrics['tau_std'][-1]:.4f}\n")
                if attention_metrics['tau_min']:
                    f.write(f"  Min:   {attention_metrics['tau_min'][-1]:.4f}\n")
                if attention_metrics['tau_max']:
                    f.write(f"  Max:   {attention_metrics['tau_max'][-1]:.4f}\n")
                if attention_metrics['tau_at_upper']:
                    f.write(f"  Upper: {attention_metrics['tau_at_upper'][-1]:.1f}%\n")
                if attention_metrics['tau_at_lower']:
                    f.write(f"  Lower: {attention_metrics['tau_at_lower'][-1]:.1f}%\n")
            
            f.write("\n--- RL Performance (Best Return) ---\n")
            f.write(f"  Newton Method:      {newton_best:.2f}\n")
            f.write(f"  Attention Network:  {attention_best:.2f}\n")
        
        print(f"✓ Saved summary to: {summary_file}")
    
    def save_metrics_json(self, newton_metrics, attention_metrics):
        """Save metrics to JSON for further analysis"""
        data = {
            'experiment_info': {
                'name': self.output_dir.name,
                'date': datetime.now().isoformat(),
                'env_id': self.env_id,
                'seed': self.seed,
                'n_rm_epochs': self.n_rm_epochs,
                'tau_max': self.tau_max,
                'rho': self.rho,
                'rl_steps': self.rl_steps
            },
            'newton_metrics': newton_metrics,
            'attention_metrics': attention_metrics
        }
        
        json_file = self.output_dir / 'metrics.json'
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Saved metrics to: {json_file}")
    
    def run_full_comparison(self):
        """Run complete comparison experiment"""
        # Run Newton method
        success1 = self.run_experiment("newton_method", use_attention=False)
        if not success1:
            print("\n✗ Newton method failed. Aborting comparison.")
            return False
        
        print("\n" + "="*70)
        print("Newton method completed. Starting Attention method...")
        print("="*70 + "\n")
        time.sleep(2)
        
        # Run Attention method
        success2 = self.run_experiment("attention_method", use_attention=True)
        if not success2:
            print("\n✗ Attention method failed.")
            return False
        
        print("\n" + "="*70)
        print("Both experiments completed. Analyzing results...")
        print("="*70 + "\n")
        
        # Parse logs
        newton_log = self.output_dir / "newton_method.log"
        attention_log = self.output_dir / "attention_method.log"
        
        print("Parsing Newton method log...")
        newton_metrics = self.parse_log_file(newton_log)
        
        print("Parsing Attention method log...")
        attention_metrics = self.parse_log_file(attention_log)
        
        # Generate outputs
        print("\nGenerating comparison plots...")
        self.generate_comparison_plots(newton_metrics, attention_metrics)
        
        print("Saving metrics...")
        self.save_metrics_json(newton_metrics, attention_metrics)
        
        print("Generating summary...")
        self.print_summary(newton_metrics, attention_metrics)
        
        print(f"\n{'='*70}")
        print("EXPERIMENT COMPLETE!")
        print(f"{'='*70}")
        print(f"\nAll results saved to: {self.output_dir}")
        print(f"  - Logs: newton_method.log, attention_method.log")
        print(f"  - Plots: comparison_plots.png")
        print(f"  - Data: metrics.json")
        print(f"  - Summary: summary.txt")
        print(f"\n{'='*70}\n")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Run automated comparison between Newton and Attention APS methods'
    )
    parser.add_argument('--env_id', type=str, default='AntBulletEnv-v0',
                       help='Environment ID (default: AntBulletEnv-v0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--n_rm_epochs', type=int, default=5,
                       help='Number of RM training epochs (default: 5)')
    parser.add_argument('--tau_max', type=float, default=1.0,
                       help='Maximum tau value (default: 1.0)')
    parser.add_argument('--rho', type=float, default=0.1,
                       help='Regularization strength (default: 0.1)')
    parser.add_argument('--rl_steps', type=int, default=1000000,
                       help='Total RL training steps (default: 1000000)')
    
    args = parser.parse_args()
    
    # Create runner and execute
    runner = ExperimentRunner(
        env_id=args.env_id,
        seed=args.seed,
        n_rm_epochs=args.n_rm_epochs,
        tau_max=args.tau_max,
        rho=args.rho,
        rl_steps=args.rl_steps
    )
    
    success = runner.run_full_comparison()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
