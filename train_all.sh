#!/bin/bash

echo "ImageNet-lightweight: Training all models"
echo "=========================================="

DATA_PATH="data/imagenet100"
GPU=0

if [ ! -d "$DATA_PATH" ]; then
    echo "Error: Dataset not found at $DATA_PATH"
    echo "Please prepare the dataset first using:"
    echo "  python scripts/prepare_data.py --imagenet-path /path/to/imagenet --output $DATA_PATH --num-classes 100"
    exit 1
fi

MODELS=("mobilenetv3_large" "mobilenetv3_small" "efficientnetv2_s" "mobilevit_s" "mobilevit_xs" "resnet50")

for model in "${MODELS[@]}"; do
    echo ""
    echo "Training $model..."
    echo "----------------------------------------"
    
    python train.py \
        --config "configs/${model}.yaml" \
        --data-path "$DATA_PATH" \
        --gpu $GPU
    
    if [ $? -eq 0 ]; then
        echo "$model training completed successfully"
    else
        echo "Error: $model training failed"
    fi
done

echo ""
echo "All models trained!"
echo "Running benchmark..."

python benchmark.py \
    --configs-dir configs \
    --checkpoints-dir experiments \
    --data-path "$DATA_PATH" \
    --output results/benchmark_results.json \
    --gpu $GPU

echo ""
echo "Generating plots..."

for model in "${MODELS[@]}"; do
    python plot.py \
        --logs "experiments/${model}/logs" \
        --output "plots/${model}"
done

python plot.py \
    --results results/benchmark_results.json \
    --output plots

echo ""
echo "Done! Check the results/ and plots/ directories."
