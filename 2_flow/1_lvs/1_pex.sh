########################################################
###### Magic netlist extraction
########################################################

export MAGIC=magic
export PDKPATH=/path/to/1_input/pdk/sky130A ; 
export MAGTYPE=mag

MAGTYPE=$MAGTYPE $MAGIC -dnull -noconsole -rcfile $PDKPATH/libs.tech/magic/sky130A.magicrc  << EOF

crashbackups stop

drc on
drc check

gds readonly true
gds flatten true
gds rescale true

tech unlock *
cif istyle sky130(vendor)
gds read $1
load ${1%.gds} -dereference
select top cell

extract do local
extract do resistance
extract all

extresist all

ext2sim labels on
ext2sim

ext2spice extresist off
ext2spice cthresh 0
ext2spice format ngspice

ext2spice -o ./gds-extracted.spice

ext2spice extresist off
ext2spice lvs
ext2spice -o ./gds-extracted_lvs.spice

EOF
#\rm -rf *.ext
#mv *.ext ./results
#mv *gds-extracted* results