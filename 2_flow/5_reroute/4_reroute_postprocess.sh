#!/bin/bash

# Assign1command-line arguments to variables
netlist_path=$1
cell_to_move=$2
action=$3
episode=$4
acc_transforms=$5
run_path=$6
f_episode=$7
mode=$8

# Validate input arguments
if [[ -z $netlist_path ]]; then
    echo "Error: Missing argument netlist_path"
    exit 1
fi

rm $netlist_path/*.ext

# Extract file name and directory path from netlist_path
file_name=$(basename "$netlist_path")

#subdir=$run_path/"iter_"$f_episode"_"$file_name/$file_name"_"$episode
if [ "$mode" = "dataset" ]; then
    subdir="$run_path/${file_name}/variants/iter_${f_episode}"
else
    subdir="$run_path/iter_${f_episode}_${file_name}/${file_name}_${episode}"
fi

new_netlist_path=$(echo "$netlist_path" | awk -F'/filtered_netlists' '{print $1}')

task_postprocess="python 2_flow/4_postprocess/0_write_out_reroute.py $new_netlist_path/simulations/$file_name $subdir/"

echo $task_postprocess

eval "$task_postprocess"
if [ $? -ne 0 ]; then
    echo "Error in postprocessing"
    exit 1
fi

exit 0