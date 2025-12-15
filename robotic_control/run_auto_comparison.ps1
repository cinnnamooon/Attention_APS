# Automated comparison script wrapper
# This script activates conda environment and runs the Python comparison script

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Automated APS Comparison Experiment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check if conda environment exists
$condaEnv = "robotic_control"
Write-Host "`nActivating conda environment: $condaEnv" -ForegroundColor Yellow

# Activate conda and run experiment
conda activate $condaEnv

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to activate conda environment '$condaEnv'" -ForegroundColor Red
    Write-Host "Please create the environment first or check the name." -ForegroundColor Red
    exit 1
}

Write-Host "✓ Environment activated" -ForegroundColor Green

# Run the automated comparison
Write-Host "`nStarting automated comparison..." -ForegroundColor Yellow
Write-Host "This will run both Newton and Attention methods sequentially." -ForegroundColor Yellow
Write-Host "Estimated time: 2-3 hours" -ForegroundColor Yellow
Write-Host ""

python run_auto_comparison.py @args

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nError: Comparison failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Comparison Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nCheck the experiments/auto_comparison_* folder for results." -ForegroundColor White
