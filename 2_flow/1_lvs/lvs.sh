#!/bin/bash

SP_DIR=$1

if [[ -z $SP_DIR ]]; then
    echo "Missing argument SP_DIR"
    exit 1
fi

SP_NL=$(basename $SP_DIR)
CUR_DIR=$(pwd)

echo "I received SP_DIR: $SP_DIR SP_NL: $SP_NL"

# CUR_DIR=$(pwd)
# item_itr=0

python "2_flow/1_lvs/0_sp2lvs.py" "$SP_DIR/$SP_NL.sp" "$SP_DIR/$SP_NL.lvs.spice"
cd $SP_DIR && $CUR_DIR/2_flow/1_lvs/1_pex.sh "${SP_NL^^}_0.gds" && cd $CUR_DIR
cd $SP_DIR && netgen -batch lvs "gds-extracted_lvs.spice ${SP_NL^^}_0" "$SP_NL.lvs.spice $SP_NL" $CUR_DIR/2_flow/1_lvs/2_netgen_setup.tcl && cd $CUR_DIR

# # echo "Processing: $item"
# python3 "$CUR_DIR/2_flow/2_lvs/0_sp2lvs.py" "$CUR_DIR/${item}/$(basename $item).sp" "$CUR_DIR/${item}/$(basename $item).lvs.spice"
# align_name=$(basename $item)
# cd $CUR_DIR/$item && $CUR_DIR/2_flow/2_lvs/1_pex.sh "$CUR_DIR/${item}/${align_name^^}_0.gds"
# echo "RUNNING FROM $(pwd)"
# netgen -batch lvs "gds-extracted_lvs.spice ${align_name^^}_0" "$CUR_DIR/${item}/$(basename $item).lvs.spice $(basename $item)" $CUR_DIR/2_flow/2_lvs/2_netgen_setup.tcl && cd ../

# mkdir "$CUR_DIR/$SP_DIR/filtered_netlists/"
# python3 "$CUR_DIR/2_flow/2_lvs/3_validate.py" "$CUR_DIR/$SP_DIR/unfiltered_netlists/" "$CUR_DIR/$SP_DIR"/filtered_netlists/
