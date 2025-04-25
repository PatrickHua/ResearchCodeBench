#!/bin/bash

# Set execution directory to the script's directory
cd "$(dirname "$0")"

# Default values
BATCH_SIZE=20
RESUME_DIR=""
ANALYZE_ONLY=false
MAX_RETRIES=10
OPTIMIZE_BATCH=false

# Process command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --resume-dir)
      RESUME_DIR="$2"
      shift 2
      ;;
    --analyze-only)
      ANALYZE_ONLY=true
      shift
      ;;
    --max-retries)
      MAX_RETRIES="$2"
      shift 2
      ;;
    --optimize-batch)
      OPTIMIZE_BATCH=true
      shift
      ;;
    *)
      # Skip unknown option
      shift
      ;;
  esac
done

# Function to build the command
build_command() {
  local resume_dir=$1
  local batch_size=$2
  
  CMD="python -m core.analysis.classify_errors"
  
  # Add required arguments
  CMD+=" --resume_from_ckpt_dir outputs/20llms_greedy/2025-04-24-13-55-17/"
  CMD+=" --data_folder pset/"
  CMD+=" --problems \"advantage-alignment\" \"Diff-Transformer\" \"DiffusionDPO\" \"DyT\" \"eomt\" \"fractalgen\" \"GMFlow\" \"GPS\" \"grid-cell-conformal-isometry\" \"hyla\" \"LEN\" \"llm-sci-use\" \"minp\" \"OptimalSteps\" \"REPA-E\" \"schedule_free\" \"semanticist\" \"SISS\" \"TabDiff\" \"Tanh-Init\""
  CMD+=" --output_dir outputs/error_classification/"
  CMD+=" --llm_types GEMINI_2_0_FLASH GPT_4O_2024_08_06 GPT_4O_MINI O3_MINI_HIGH DEEPSEEK_R1 O1_HIGH GPT_4_1 GPT_4_1_MINI GPT_4_1_NANO O3_HIGH CLAUDE_3_5_SONNET_2024_10_22 CLAUDE_3_7_SONNET_2025_02_19 GROK_3_BETA GEMINI_2_5_PRO_PREVIEW_03_25"
  
  # Note: We're not adding custom arguments via command line since they're not supported
  # Instead, modify core/analysis/classify_errors.py directly to use these values
  
  if [ "$resume_dir" != "" ]; then
    # The resume_dir argument is handled in the code, so we pass it as an environment variable
    CMD="RESUME_DIR=\"$resume_dir\" $CMD"
  fi
  
  if [ "$ANALYZE_ONLY" = true ]; then
    # The analyze_only argument is handled in the code, so we pass it as an environment variable
    CMD="ANALYZE_ONLY=true $CMD"
  fi
  
  # Pass batch size as an environment variable
  CMD="BATCH_SIZE=$batch_size $CMD"
  
  echo "$CMD"
}

# Function to run command and measure time
run_with_timing() {
  local cmd="$1"
  local max_time="$2"  # Maximum time to run in seconds
  
  # Start timing
  start_time=$(date +%s)
  
  # Run the command with timeout
  timeout $max_time bash -c "$cmd" &
  pid=$!
  
  # Wait for command to finish or timeout
  wait $pid
  exit_code=$?
  
  # Get end time
  end_time=$(date +%s)
  elapsed=$((end_time - start_time))
  
  echo "$elapsed"  # Return elapsed time
  return $exit_code
}

# Function to optimize batch size
optimize_batch_size() {
  echo "=== Batch Size Optimization ==="
  echo "Running short tests with different batch sizes to find optimal speed"
  
  # Test batch sizes: 5, 10, 20, 40, 80
  batch_sizes=(5 10 20 40 80)
  best_batch_size=20
  best_time=999999
  
  for size in "${batch_sizes[@]}"; do
    echo ""
    echo "Testing batch size: $size"
    cmd=$(build_command "" $size)
    
    # Run for max 2 minutes to get a sense of speed
    elapsed=$(run_with_timing "$cmd" 120)
    
    if [ $? -eq 124 ]; then
      # Command timed out - we count this as full 120 seconds
      elapsed=120
      echo "  Test timed out after 120 seconds"
    fi
    
    # Calculate errors per second based on first batch
    # This is approximate and assumes first batch completed
    echo "  Time elapsed: $elapsed seconds"
    
    # If time is better than previous best, update best
    if [ $elapsed -lt $best_time ]; then
      best_time=$elapsed
      best_batch_size=$size
      echo "  New best batch size: $size (time: $elapsed seconds)"
    fi
    
    # Clean up any processes before next test
    pkill -f "core.analysis.classify_errors" 2>/dev/null
    sleep 5
  done
  
  echo ""
  echo "Optimization complete!"
  echo "Best batch size: $best_batch_size"
  echo "Setting batch size to $best_batch_size for full run"
  
  return $best_batch_size
}

# Display initial info
echo "=== Error Classification Runner ==="
echo "This script will run the error classification until it completes successfully."

# Run batch size optimization if requested
if [ "$OPTIMIZE_BATCH" = true ]; then
  optimize_batch_size
  BATCH_SIZE=$?
fi

echo "- Will process errors in batches of $BATCH_SIZE (passed as environment variable)"
echo "- Will retry up to $MAX_RETRIES times on failure"
if [ "$RESUME_DIR" != "" ]; then
  echo "- Starting from directory: $RESUME_DIR"
fi
if [ "$ANALYZE_ONLY" = true ]; then
  echo "- Only analyzing existing results without running classification"
fi
echo ""
echo "Press Ctrl+C to cancel or wait 3 seconds to continue..."
sleep 3

# Run until successful
RETRY_COUNT=0
SUCCESS=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ] && [ "$SUCCESS" = false ]; do
  # If this is a retry and we have a latest directory, use it for resuming
  if [ $RETRY_COUNT -gt 0 ]; then
    echo ""
    echo "Retry #$RETRY_COUNT of $MAX_RETRIES"
    
    # Try to find latest directory to resume from
    if [ -L "outputs/error_classification/latest" ]; then
      RESUME_DIR=$(readlink -f "outputs/error_classification/latest")
      echo "Resuming from latest directory: $RESUME_DIR"
    elif [ "$RESUME_DIR" = "" ] && [ -d "outputs/error_classification" ]; then
      # Find most recent directory by timestamp
      LATEST_DIR=$(find "outputs/error_classification" -maxdepth 1 -type d -not -name "error_classification" | sort -r | head -n 1)
      if [ "$LATEST_DIR" != "" ]; then
        RESUME_DIR="$LATEST_DIR"
        echo "Resuming from most recent directory: $RESUME_DIR"
      fi
    fi
  fi
  
  # Build the command with the current resume directory
  CMD=$(build_command "$RESUME_DIR" $BATCH_SIZE)
  
  echo ""
  echo "Running command: $CMD"
  echo "Timestamp: $(date)"
  echo "---------------------------------------------"
  
  # Run the command
  eval $CMD
  EXIT_CODE=$?
  
  if [ $EXIT_CODE -eq 0 ]; then
    SUCCESS=true
    echo "---------------------------------------------"
    echo "Command completed successfully!"
  else
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "---------------------------------------------"
    echo "Command failed with exit code $EXIT_CODE."
    
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
      echo "Waiting 30 seconds before retrying..."
      sleep 30
    else
      echo "Maximum retry attempts reached. Giving up."
    fi
  fi
done

if [ "$SUCCESS" = true ]; then
  echo ""
  echo "=== Error Classification Complete ==="
  echo "Results saved to: outputs/error_classification/latest"
  echo "You can view visualizations in: outputs/error_classification/latest/error_visualizations"
else
  echo ""
  echo "=== Error Classification Failed ==="
  echo "Please check the error messages above and try again."
  echo "You might be able to resume from: outputs/error_classification/latest"
  exit 1
fi 