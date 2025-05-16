import argparse
import os
import sys
import shutil


def process_file(dir_name, variant_dir, cmp_file="comp.out", check_phrase="Final result: Circuits match uniquely"):
    passed=False

    with open(f"{variant_dir}/{cmp_file}", "r") as file:
        content = file.read()
        if check_phrase in content:
            passed=True
            variant_name = variant_dir.split("/")[-1]
            shutil.copytree(f"{dir_name}/unfiltered_netlists/{variant_name}", f"{dir_name}/filtered_netlists/{variant_name}", dirs_exist_ok=True)

    return passed

def main():
    # Set up argparse for command-line arguments
    parser = argparse.ArgumentParser(description="Process a SPECTRE netlist file and prepares it for LVS.")
    parser.add_argument("dir_name", help="The input SPECTRE netlists")
    parser.add_argument("variant_dir", help="The input SPECTRE netlists")


    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the process_file function with the provided file paths
    res = process_file(args.dir_name, args.variant_dir)

    if res:
        sys.exit(0)
    else:
        sys.exit(1)

# Run the script if it is executed directly
if __name__ == "__main__":
    main()
