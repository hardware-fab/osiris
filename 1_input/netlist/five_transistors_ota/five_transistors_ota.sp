.subckt five_transistors_ota GND VDD VOUT VINN VINP VBIAS
M0 source VBIAS GND GND sky130_fd_pr__nfet_01v8 l=150e-9 w=10.5e-7 nf=ZZZ
M1 VOUT VINN source GND sky130_fd_pr__nfet_01v8 l=150e-9 w=10.5e-7 nf=ZZZ
M2 net8 VINP source GND sky130_fd_pr__nfet_01v8 l=150e-9 w=10.5e-7 nf=ZZZ
M3 VOUT net8 VDD VDD sky130_fd_pr__pfet_01v8 l=150e-9 w=10.5e-7 nf=ZZZ
M4 net8 net8 VDD VDD sky130_fd_pr__pfet_01v8 l=150e-9 w=10.5e-7 nf=ZZZ
.ends five_transistors_ota