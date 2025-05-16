
import re
import argparse

# Dictionary to map component types to new models
model_map = {
    'nch':     'sky130_fd_pr__nfet_01v8',
    'pch':     'sky130_fd_pr__pfet_01v8',
    'nch_lvt': 'sky130_fd_pr__nfet_01v8_lvt',
    'pch_lvt': 'sky130_fd_pr__pfet_01v8_lvt',
    'cap':     'sky130_fd_pr__cap_mim_m3_1',
    'rpoly':   'sky130_fd_pr__res_generic_l1',

    'ne':    'sky130_fd_pr__nfet_01v8',
    'pe':    'sky130_fd_pr__pfet_01v8',
    'nel':   'sky130_fd_pr__nfet_01v8_lvt',
    'pel':   'sky130_fd_pr__pfet_01v8_lvt',
    'cmm5t': 'sky130_fd_pr__cap_mim_m3_1',
    'rnp1':  'sky130_fd_pr__res_generic_l1',

    'sky130_fd_pr__nfet_01v8':      'sky130_fd_pr__nfet_01v8',
    'sky130_fd_pr__pfet_01v8':      'sky130_fd_pr__pfet_01v8',
    'sky130_fd_pr__nfet_01v8_lvt':  'sky130_fd_pr__nfet_01v8_lvt',
    'sky130_fd_pr__pfet_01v8_lvt':  'sky130_fd_pr__pfet_01v8_lvt',
    'sky130_fd_pr__cap_mim_m3_1':   'sky130_fd_pr__cap_mim_m3_1',
    'sky130_fd_pr__res_generic_l1': 'sky130_fd_pr__res_generic_l1',
    
    'resistor': 'sky130_fd_pr__res_generic_l1',
}

def process_file(input_file_path, output_file_path):
    # Output list to store the transformed lines
    output_lines = []

    with open(input_file_path, "r") as file:
        netlist = file.read().split("\n")
        output_lines.append("* Created from sp2lvs.py\n")

        mos_n = 0
        for line in netlist:
            line = line.strip()
            if line.startswith("M"):
                mos_n+=1

        # Process each line
        for line in netlist:
            line = line.strip()
            
            # Skip the subckt and ends lines, handled separately
            if line.startswith(".subckt"):
                output_lines.append(line)
            elif line.startswith(".ends"):
                output_lines.append(".ends\n")
                    
            if line.startswith("M"):
                match = re.match(
                    r"(M\d+)\s+([\S ]+)\s+(\S+)\s+l=([\d\.e-]+[unp]?)\s+w=([\d\.e-]+[unp]?)\s*(?:nf=(\d+))?",
                    line
                )
            elif line.startswith("C"):
                match = re.match(
                    r"(C\d+)\s+([\S ]+)\s+(\S+)\s+l=([\d\.e-]+[unp]?)\s+w=([\d\.e-]+[unp]?)",
                    line
                )
            elif line.startswith("R"):
                match = re.match(
                    r"(R\d+)\s+([\S ]+)\s+(\S+)\s*r=([\d\.]+[kmunp]?)",
                    line
                )
            else:
                continue

            def force_format(num):
                """Formats any number to always end with e-6 notation."""
                scaled_num = num * 1e6  # Scale the number to match e-6
                return f"{scaled_num:.3f}e-6"  # Format with 6 decimal places

            if match:
                # Extract the relevant parts
                print(match.groups())
                instance_name = match.groups()[0]
                if instance_name.startswith("M"):
                    _, nets, model, l, w, nf = match.groups()
                    new_model = model_map.get(model, model)
                    len_rebased = float(l)
                    width_rebased = float(w)
                    # width_rebased = float(w)/float(nf)
                    for nf_itr in range(0, int(nf)):
                        new_line = f"{instance_name.replace('M', 'X')}_{nf_itr} {nets} {new_model} l={force_format(len_rebased)} w={force_format(width_rebased)}"
                        output_lines.append(new_line)                
                elif instance_name.startswith("C"):
                    _, nets, model, l, w = match.groups()
                    new_instance_name = f"X{mos_n}"
                    new_model = model_map.get(model, model)
                    len_rebased = float(l)
                    width_rebased = float(w)
                    new_line = f"{new_instance_name} {nets} {new_model} l={force_format(len_rebased)} w={force_format(width_rebased)}"
                    output_lines.append(new_line)
                    mos_n+=1
                elif instance_name.startswith("R"):
                    _, nets, model, r = match.groups()
                    new_model = model_map.get(model, model)
                    new_line = f"{instance_name} {nets} {new_model} r={r}"
                    output_lines.append(new_line)                
                else:
                    continue

        # Join output lines into the final netlist format
        #transformed_netlist = "\n".join(output_lines)

        # Output the transformed netlist
        with open(output_file_path, "w") as output_file:
            for line in output_lines:
                output_file.write(line + "\n")


# Main function to parse arguments and run the script
def main():
    # Set up argparse for command-line arguments
    parser = argparse.ArgumentParser(description="Process a SPECTRE netlist file and prepares it for LVS.")
    parser.add_argument("input_file", help="The input SPECTRE netlist file")
    parser.add_argument("output_file", help="The output LVS netlist file")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the process_file function with the provided file paths
    process_file(args.input_file, args.output_file)

# Run the script if it is executed directly
if __name__ == "__main__":
    main()
