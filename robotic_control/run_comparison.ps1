# Comparison experiments between Newton method and Attention network method
# This script runs both methods with identical parameters for fair comparison

$env_id = "AntBulletEnv-v0"
$seed = 42
$n_rm_epochs = 5
$tau_max = 1.0
$rho = 0.1

# Create output directory
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$output_dir = "experiments/comparison_$timestamp"
New-Item -ItemType Directory -Force -Path $output_dir | Out-Null

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Running Comparison Experiments" -ForegroundColor Cyan
Write-Host "Environment: $env_id" -ForegroundColor Cyan
Write-Host "Seed: $seed" -ForegroundColor Cyan
Write-Host "RM Epochs: $n_rm_epochs" -ForegroundColor Cyan
Write-Host "Tau Max: $tau_max" -ForegroundColor Cyan
Write-Host "Rho: $rho" -ForegroundColor Cyan
Write-Host "Output: $output_dir" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Experiment 1: Original Newton Method
Write-Host "`n[1/2] Running Newton Method (Original APS)..." -ForegroundColor Yellow
python train_rlhf.py `
    --env_id $env_id `
    --seed $seed `
    --enable_individualized_tau `
    --n_rm_epochs $n_rm_epochs `
    --tau_max $tau_max `
    --rho $rho `
    --output_path "$output_dir/newton_method" `
    | Tee-Object -FilePath "$output_dir/newton_method.log"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Newton method training failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Newton method completed" -ForegroundColor Green

# Experiment 2: Attention Network Method
Write-Host "`n[2/2] Running Attention Network Method..." -ForegroundColor Yellow
python train_rlhf.py `
    --env_id $env_id `
    --seed $seed `
    --enable_individualized_tau `
    --use_attention_tau `
    --n_rm_epochs $n_rm_epochs `
    --tau_max $tau_max `
    --rho $rho `
    --attention_hidden_dim 128 `
    --attention_num_heads 4 `
    --attention_num_layers 2 `
    --attention_lr 0.0001 `
    --train_attention_freq 5 `
    --output_path "$output_dir/attention_method" `
    | Tee-Object -FilePath "$output_dir/attention_method.log"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Attention method training failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Attention method completed" -ForegroundColor Green

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Comparison Experiments Completed!" -ForegroundColor Cyan
Write-Host "Results saved in: $output_dir" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Check logs: $output_dir/*.log" -ForegroundColor White
Write-Host "2. Compare results:" -ForegroundColor White
Write-Host "   - Newton method: $output_dir/newton_method/" -ForegroundColor White
Write-Host "   - Attention method: $output_dir/attention_method/" -ForegroundColor White
Write-Host "3. Run analysis script (if available)" -ForegroundColor White
