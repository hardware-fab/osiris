.subckt miller_ota INM INP GND VDD OUT
C0 net12 OUT sky130_fd_pr__cap_mim_m3_1 l=24.0e-6 w=24.0e-6
R0 out1 net12 resistor r=18k
M0 net10 INM net11 GND sky130_fd_pr__nfet_01v8 l=150.0e-9 w=21.0e-7 nf=ZZZ
M2 out1 INP net11 GND sky130_fd_pr__nfet_01v8 l=150.0e-9 w=15.75e-7 nf=ZZZ
M6 OUT vbias GND GND sky130_fd_pr__nfet_01v8 l=150.0e-9 w=21.0e-7 nf=ZZZ
M1 net11 vbias GND GND sky130_fd_pr__nfet_01v8 l=150.0e-9 w=10.5e-7 nf=ZZZ
M10 vbias vbias GND GND sky130_fd_pr__nfet_01v8 l=150.0e-9 w=10.5e-7 nf=ZZZ
M5 OUT out1  VDD VDD sky130_fd_pr__pfet_01v8 l=150.0e-9 w=5.25e-7 nf=ZZZ
M3 out1 net10 VDD VDD sky130_fd_pr__pfet_01v8 l=150.0e-9 w=21.0e-7 nf=ZZZ
M4 net10 net10 VDD VDD sky130_fd_pr__pfet_01v8 l=150.0e-9 w=21.0e-7 nf=ZZZ
M8 vbias vbias VDD VDD sky130_fd_pr__pfet_01v8 l=150.0e-9 w=5.25e-7 nf=ZZZ
.ends miller_ota