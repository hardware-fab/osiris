#!/bin/bash

RESULTS_DIR=$1
VARIANT_PATH=$2
OUTPUT_DIR=$3

if [[ -z $OUTPUT_DIR ]]; then
    echo "Missing argument OUTPUT_DIR"
    exit 1
fi

if [[ -z $RESULTS_DIR ]]; then
    echo "Missing argument RESULTS_DIR"
    exit 1
fi

VARIANT_NAME=$(basename $VARIANT_PATH)

python 2_flow/4_postprocess/0_write_out.py "$RESULTS_DIR/simulations/$VARIANT_NAME" "$OUTPUT_DIR"