#!/bin/bash
CONFIG_DIR="$1"

for cfg in "$CONFIG_DIR"/*.yaml; do
  echo "Running with config: $cfg"
  python main.py --config="$cfg"
done
