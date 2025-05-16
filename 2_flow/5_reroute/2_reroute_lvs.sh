#!/bin/bash

# Assign1command-line arguments to variables
netlist_path=$1
cell_to_move=$2
action=$3
episode=$4

# Validate input arguments
if [[ -z $netlist_path ]]; then
    echo "Error: Missing argument netlist_path"
    exit 1
fi

rm $netlist_path/*.ext

# Extract file name and directory path from netlist_path
file_name=$(basename "$netlist_path")

output_dir=3_output/$file_name

if [[ ! -d $output_dir ]]; then
    mkdir 3_output
    mkdir $output_dir
fi

variants_dir=$output_dir/variants
if [[ ! -d $variants_dir ]]; then
    mkdir $variants_dir
fi

subdir=$variants_dir/$file_name"_"$episode

task_lvs="2_flow/1_lvs/lvs.sh $netlist_path"
lvs_output=$($task_lvs 2>&1)
lvs_return_code=$?

# Check if the command failed
if [ $lvs_return_code -ne 0 ]; then
    echo "❌ Error in LVS check."
    exit 1
fi

if echo "$lvs_output" | grep -q "Netlists do not match"; then
    echo "❌ Netlists do not match."
    exit 1
fi

if echo "$lvs_output" | grep -q "Top level cell failed pin matching"; then
    echo "❌ Pin matching failed."
    exit 1
fi

exit 0

wait  # Wait for all background processes to finish
