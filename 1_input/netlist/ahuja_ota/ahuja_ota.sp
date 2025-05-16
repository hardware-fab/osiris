.subckt ahuja_ota INM INP GND VDD OUT
M7 VDD out1 net1 GND sky130_fd_pr__nfet_01v8 l=150.0e-9 w=5.25e-7 nf=ZZZ
M10 vbias vbias GND GND sky130_fd_pr__nfet_01v8 l=150.0e-9 w=5.25e-7 nf=ZZZ
M11 vcas vcas vbias GND sky130_fd_pr__nfet_01v8 l=150.0e-9 w=10.5e-7 nf=ZZZ
M9 net1 vbias GND GND sky130_fd_pr__nfet_01v8 l=150.0e-9 w=15.75e-7 nf=ZZZ
M6 OUT net1 GND GND sky130_fd_pr__nfet_01v8 l=150.0e-9 w=5.25e-7 nf=ZZZ
M13 out1 vcas net7 GND sky130_fd_pr__nfet_01v8 l=150.0e-9 w=10.5e-7 nf=ZZZ
M12 net18 vcas net16 GND sky130_fd_pr__nfet_01v8 l=150.0e-9 w=10.5e-7 nf=ZZZ
M0 net16 INM net11 GND sky130_fd_pr__nfet_01v8 l=150.0e-9 w=21.0e-7 nf=ZZZ
M2 net7 INP net11 GND sky130_fd_pr__nfet_01v8 l=150.0e-9 w=21.0e-7 nf=ZZZ
M1 net11 vbias GND GND sky130_fd_pr__nfet_01v8 l=150.0e-9 w=5.25e-7 nf=ZZZ
C0 net7 OUT sky130_fd_pr__cap_mim_m3_1 l=25.0e-6 w=25.0e-6
M8 vcas vcas VDD VDD sky130_fd_pr__pfet_01v8 l=150.0e-9 w=10.5e-7 nf=ZZZ
M5 OUT out1 VDD VDD sky130_fd_pr__pfet_01v8 l=150.0e-9 w=15.75e-7 nf=ZZZ
M3 out1 net18 VDD VDD sky130_fd_pr__pfet_01v8 l=150.0e-9 w=21.0e-7 nf=ZZZ
M4 net18 net18 VDD VDD sky130_fd_pr__pfet_01v8 l=150.0e-9 w=21.0e-7 nf=ZZZ
.ends ahuja_ota