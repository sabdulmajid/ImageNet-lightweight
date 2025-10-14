# ImageNet-lightweight: Training all models
# ==========================================

$DATA_PATH = "data/imagenet100"
$GPU = 0

if (-not (Test-Path $DATA_PATH)) {
    Write-Host "Error: Dataset not found at $DATA_PATH" -ForegroundColor Red
    Write-Host "Please prepare the dataset first using:"
    Write-Host "  python scripts/prepare_data.py --imagenet-path /path/to/imagenet --output $DATA_PATH --num-classes 100"
    exit 1
}

$MODELS = @("mobilenetv3_large", "mobilenetv3_small", "efficientnetv2_s", "mobilevit_s", "mobilevit_xs", "resnet50")

foreach ($model in $MODELS) {
    Write-Host ""
    Write-Host "Training $model..." -ForegroundColor Cyan
    Write-Host "----------------------------------------"
    
    python train.py `
        --config "configs/$model.yaml" `
        --data-path $DATA_PATH `
        --gpu $GPU
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "$model training completed successfully" -ForegroundColor Green
    } else {
        Write-Host "Error: $model training failed" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "All models trained!" -ForegroundColor Green
Write-Host "Running benchmark..."

python benchmark.py `
    --configs-dir configs `
    --checkpoints-dir experiments `
    --data-path $DATA_PATH `
    --output results/benchmark_results.json `
    --gpu $GPU

Write-Host ""
Write-Host "Generating plots..."

foreach ($model in $MODELS) {
    python plot.py `
        --logs "experiments/$model/logs" `
        --output "plots/$model"
}

python plot.py `
    --results results/benchmark_results.json `
    --output plots

Write-Host ""
Write-Host "Done! Check the results/ and plots/ directories." -ForegroundColor Green
