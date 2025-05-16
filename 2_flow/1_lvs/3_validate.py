import argparse
import os
import shutil

def process_file(input_nl_path, output_nl_path, cmp_file="comp.out", check_phrase="Final result: Circuits match uniquely"):
    filtered_files = list()
    for gen_nl in os.listdir(input_nl_path):
        for gen_nl_file in os.listdir(input_nl_path + "/" + gen_nl):
            if gen_nl_file == cmp_file:
                with open(f"{input_nl_path}/{gen_nl}/{gen_nl_file}", "r") as file:
                    content = file.read()
                    if check_phrase in content:
                        filtered_files.append(f"{input_nl_path}/{gen_nl}")
    for file in filtered_files:
        dir_name = file.split("/")[-1]
        shutil.copytree(file, output_nl_path + f"{dir_name}", dirs_exist_ok=True)

def main():
    # Set up argparse for command-line arguments
    parser = argparse.ArgumentParser(description="Process a SPECTRE netlist file and prepares it for LVS.")
    parser.add_argument("netlist_path", help="The input SPECTRE netlists")
    parser.add_argument("output_path", help="The output LVS netlist file")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the process_file function with the provided file paths
    process_file(args.netlist_path, args.output_path)

# Run the script if it is executed directly
if __name__ == "__main__":
    main()
