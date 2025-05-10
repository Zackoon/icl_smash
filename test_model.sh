#!/bin/bash
# Simple script to run the model responsiveness test

# Default values
MODEL_PATH="./trained_model.pt"
STATS_DIR="./model_params"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --stats-dir)
      STATS_DIR="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --model PATH       Path to the trained model (default: ./trained_model.pt)"
      echo "  --stats-dir PATH   Directory containing normalization statistics (default: ./model_params)"
      echo "  --help             Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
  echo "Error: Model file not found at $MODEL_PATH"
  exit 1
fi

# Check if stats directory exists
if [ ! -d "$STATS_DIR" ]; then
  echo "Error: Stats directory not found at $STATS_DIR"
  exit 1
fi

# Run the test
echo "Running model responsiveness test..."
echo "Model: $MODEL_PATH"
echo "Stats directory: $STATS_DIR"
echo ""

python3 test_model.py --model "$MODEL_PATH" --stats-dir "$STATS_DIR"

# Check exit code
if [ $? -eq 0 ]; then
  echo ""
  echo "Test completed successfully."
else
  echo ""
  echo "Test failed. See above for details."
fi