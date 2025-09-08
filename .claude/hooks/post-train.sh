#!/bin/bash
# Post-training hook for Khmer OCR - evaluates model and updates manifest

set -e

PROJECT_ROOT="/Users/niko/Desktop/khmer-ocr-v1"
MANIFEST="$PROJECT_ROOT/governance/manifest.json"
EVAL_REPORT="$PROJECT_ROOT/eval/report.json"

echo "🔍 Running post-training validation..."

# Check if model files exist
if [ -d "$PROJECT_ROOT/models" ]; then
    echo "✓ Model directory found"
    
    # Calculate SHA256 for model files
    echo "📊 Calculating model checksums..."
    find "$PROJECT_ROOT/models" -type f \( -name "*.pdparams" -o -name "*.pdmodel" -o -name "*.onnx" \) -exec sha256sum {} \; > "$PROJECT_ROOT/models/checksums.txt"
    
    # Run evaluation if available
    if [ -f "$PROJECT_ROOT/eval/harness.py" ]; then
        echo "🧪 Running evaluation harness..."
        cd "$PROJECT_ROOT"
        python eval/harness.py --test data/test --report "$EVAL_REPORT" || echo "⚠️ Evaluation failed"
        
        # Extract CER metrics
        if [ -f "$EVAL_REPORT" ]; then
            CER_CLEAN=$(jq -r '.cer_clean // "N/A"' "$EVAL_REPORT")
            CER_DEGRADED=$(jq -r '.cer_degraded // "N/A"' "$EVAL_REPORT")
            
            echo "📈 Performance Metrics:"
            echo "   - CER (clean): $CER_CLEAN"
            echo "   - CER (degraded): $CER_DEGRADED"
            
            # Check acceptance criteria
            if [[ "$CER_CLEAN" != "N/A" ]] && (( $(echo "$CER_CLEAN < 0.03" | bc -l) )); then
                echo "✅ Clean CER meets target (≤3%)"
            else
                echo "⚠️ Clean CER above target (>3%)"
            fi
            
            if [[ "$CER_DEGRADED" != "N/A" ]] && (( $(echo "$CER_DEGRADED < 0.10" | bc -l) )); then
                echo "✅ Degraded CER meets target (≤10%)"
            else
                echo "⚠️ Degraded CER above target (>10%)"
            fi
        fi
    fi
    
    # Update manifest
    if [ -f "$PROJECT_ROOT/ops/manifests.py" ]; then
        echo "📝 Updating manifest.json..."
        python "$PROJECT_ROOT/ops/manifests.py" --update
    fi
else
    echo "⚠️ No model directory found - skipping validation"
fi

echo "✨ Post-training hook completed"