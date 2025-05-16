.subckt ota_ff VDD VSS VP VN
M5  net4 net9 VDD  VDD sky130_fd_pr__pfet_01v8 l=150e-9 w=10.5e-7 nf=ZZZ
M39 net3 net3 VDD  VDD sky130_fd_pr__pfet_01v8 l=150e-9 w=10.5e-7 nf=ZZZ
M82 VN   VN   net4 VDD sky130_fd_pr__pfet_01v8 l=150e-9 w=10.5e-7 nf=ZZZ
M81 VSS  VP   net4 VDD sky130_fd_pr__pfet_01v8 l=150e-9 w=10.5e-7 nf=ZZZ
M40 net9 net3 VDD  VDD sky130_fd_pr__pfet_01v8 l=150e-9 w=10.5e-7 nf=ZZZ
M50 v1   v1   VSS  VSS sky130_fd_pr__nfet_01v8 l=150e-9 w=10.5e-7 nf=ZZZ
M36 net6 v1   VSS  VSS sky130_fd_pr__nfet_01v8 l=150e-9 w=10.5e-7 nf=ZZZ
M38 net3 VN   net6 VSS sky130_fd_pr__nfet_01v8 l=150e-9 w=10.5e-7 nf=ZZZ
M35 VDD  VP   net7 VSS sky130_fd_pr__nfet_01v8 l=150e-9 w=10.5e-7 nf=ZZZ
M34 VN   VN   net7 VSS sky130_fd_pr__nfet_01v8 l=150e-9 w=10.5e-7 nf=ZZZ
M6  net7 v1   VSS  VSS sky130_fd_pr__nfet_01v8 l=150e-9 w=10.5e-7 nf=ZZZ
M37 net9 VP   net6 VSS sky130_fd_pr__nfet_01v8 l=150e-9 w=10.5e-7 nf=ZZZ
M1  v1   v1   VDD  VDD sky130_fd_pr__pfet_01v8 l=150e-9 w=10.5e-7 nf=ZZZ
.ends