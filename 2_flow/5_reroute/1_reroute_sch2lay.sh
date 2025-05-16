#!/bin/bash

# Assign1command-line arguments to variables
netlist_path=$1
cell_to_move=$2
action=$3
episode=$4
accumulatd_transforms=$5

# Validate input arguments
if [[ -z $netlist_path ]]; then
    echo "Error: Missing argument netlist_path"
    exit 1
fi

rm -rf $netlist_path/*.ext

# Extract file name and directory path from netlist_path
file_name=$(basename "$netlist_path")

task_rr="schematic2layout $netlist_path -p /path/to/0_install/pdk/SKY130_PDK \
        -w $netlist_path --cell_to_move $cell_to_move --transformation \"$accumulatd_transforms\" --flow_start=4_reroute"

eval "$task_rr"

if [ $? -ne 0 ]; then
    echo "Error in schematic2layout"
    exit 1
fi

exit 0