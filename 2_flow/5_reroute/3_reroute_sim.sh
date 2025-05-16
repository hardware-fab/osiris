#!/bin/bash

# Assign1command-line arguments to variables
netlist_path=$1
cell_to_move=$2
action=$3
episode=$4

# Validate input arguments
if [[ -z $netlist_path ]]; then
    echo "Error: Missing argument netlist_path"
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

new_netlist_path=$(echo "$netlist_path" | awk -F'/filtered_netlists' '{print $1}')
task_sim="2_flow/3_sim/sim.sh $new_netlist_path $file_name"
eval "$task_sim"
if [ $? -ne 0 ]; then
    echo "Error in simulation"
fi

exit 0

wait  # Wait for all background processes to finish
