#!/bin/bash

netlist_path=$1
netlist_variant=$2

if [[ -z $netlist_path ]]; then
    echo "Missing argument netlist_path"
    exit 1
fi

if [[ -z $netlist_variant ]]; then
    echo "Missing argument netlist_variant"
    exit 1
fi

CUR_DIR=$(pwd)

python 2_flow/3_sim/0_prepare_sim.py $netlist_path $netlist_variant
design_name=$(basename $netlist_variant)
cd $netlist_path/simulations/$design_name && ngspice "$design_name"_pre.spice && ngspice "$design_name"_post.spice && cd $CUR_DIR