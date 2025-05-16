import argparse
import numpy as np
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import gdspy
from matplotlib.backends.backend_pdf import PdfPages


class Variable:
    def __init__(self, name, var_type, idx):
        self.name = name
        self.type = var_type
        self.idx = idx
        self.samples = np.array([])

def extract_variables(lines):
    dataset = {}
    var_count = 0
    line_itr = 0

    # Locate the line containing "No. Variables:"
    while line_itr < len(lines) and "No. Variables:" not in lines[line_itr]:
        line_itr += 1

    if line_itr >= len(lines):
        raise ValueError("No. Variables: line not found in the file.")

    # Extract the number of variables safely
    try:
        var_count = int(lines[line_itr].split()[-1])
    except ValueError:
        raise ValueError(f"Could not extract variable count from line: {lines[line_itr]}")

    # Locate the "Variables:" section
    while line_itr < len(lines) and "Variables:" not in lines[line_itr]:
        line_itr += 1

    line_itr += 1  # Move to the first variable line

    # Extract variable details
    while line_itr < len(lines) and "Values:" not in lines[line_itr]:
        tokens = lines[line_itr].split()

        if len(tokens) < 3:
            line_itr += 1
            continue  # Skip invalid lines

        try:
            idx = int(tokens[0])
            name = tokens[1]
            var_type = tokens[2]

            if "(" in name and ")" in name:
                name = name.split(")")[0].split("(")[-1]

            dataset[idx] = Variable(name, var_type, idx)
        except ValueError:
            pass  # Skip lines that do not match the expected format

        line_itr += 1

    return dataset, var_count, line_itr + 1

def extract_values(lines, dataset, var_count, line_itr):
    while line_itr < len(lines):
        if lines[line_itr]:
            for var_idx in range(var_count):
                value = float(lines[line_itr].split(",")[0].split("\t")[1])
                dataset[var_idx].samples = np.append(dataset[var_idx].samples, value)
                line_itr += 1
        line_itr += 1

def write_csv(output_file_path, dataset):
    with open(output_file_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([dataset[idx].name for idx in dataset])
        csvwriter.writerows(np.array([dataset[idx].samples for idx in dataset]).T)

def process_files(res_path, out_path):
    variant_name = res_path.split("/")[-1]
    design_name = res_path.split("/")[-3]
    pre_layout_res_path = f"{res_path}/simout_pre.out" 
    post_layout_res_path = f"{res_path}/simout_post.out" 

    # Pre-layout
    with open(pre_layout_res_path, "r") as pre_layout_res:
        pre_layout_lines = pre_layout_res.read().split("\n")
    
    pre_dataset, pre_var_count, pre_line_itr = extract_variables(pre_layout_lines)
    extract_values(pre_layout_lines, pre_dataset, pre_var_count, pre_line_itr)
    # Post-layout
    with open(post_layout_res_path, "r") as post_layout_res:
        post_layout_lines = post_layout_res.read().split("\n")
    
    post_dataset, post_var_count, post_line_itr = extract_variables(post_layout_lines)
    extract_values(post_layout_lines,post_dataset, post_var_count, post_line_itr)

    dataset_dir = f"{out_path}/"

    os.makedirs(f"{dataset_dir}", exist_ok=True)

    output_path_pre = f"{dataset_dir}/{variant_name}_pre.csv"
    output_path_post = f"{dataset_dir}/{variant_name}_post.csv"

    write_csv(output_path_pre, pre_dataset)
    write_csv(output_path_post, post_dataset)

    return output_path_pre, output_path_post

def rmse(prediction, target):
    return np.sqrt(((prediction - target) ** 2).mean())

def pex_score(pre_layout, post_layout, tolerance=0.1, weight=0.2):
    n_samples = len(pre_layout)
    out_of_bounds_indexes = list()

    for sample_itr in range(0, n_samples):
        lower_bound = pre_layout[sample_itr] * (1-tolerance) 
        upper_bound = pre_layout[sample_itr] * (1+tolerance)

        if not (lower_bound <= post_layout[sample_itr] <= upper_bound):
            out_of_bounds_indexes.append(sample_itr)

    layout_rmse = rmse(pre_layout, post_layout)
    oob_weight = weight * len(out_of_bounds_indexes)

    return layout_rmse + oob_weight

def read_csv_file(filename):
    df = pd.read_csv(filename)
    frequency = df['frequency'].values
    out = df[df.keys()[-1]].values
    return frequency, out

def plot_to_pdf(pre_pair, post_pair, score, output_pdf):
    with PdfPages(output_pdf) as pdf:
        plt.figure()
        plt.plot(np.arange(len(pre_pair[0])), pre_pair[1], label='Pre layout TF', color='b')
        plt.plot(np.arange(len(post_pair[0])), post_pair[1], label='Post layout TF', color='r')
        plt.xlabel('Frequency')
        plt.ylabel('Output')
        plt.ylim(-1.5, 1.5)
        plt.title(f'PEX Score={score}')
        plt.legend()
        pdf.savefig()
        plt.close()

def save_to_npy(output_file, score):
    np.save(output_file, np.array(score))

def process_csv_files(output_path_pre, output_path_post):
    output_path = "/".join(output_path_pre.split("/")[0:-2])
    pre_frequency, pre_out = read_csv_file(output_path_pre)
    post_frequency, post_out = read_csv_file(output_path_post)

    score = pex_score(pre_out, post_out)

    plot_to_pdf((pre_frequency, pre_out), (post_frequency, post_out), score, f"{output_path}/score.pdf")

    save_to_npy(f"{output_path}/score.npy", score)

def copy_gds(results_path, output_path_pre):
    output_path = "/".join(output_path_pre.split("/")[0:-2])
    gds_path = "/".join(results_path.split("/")[0:-2])+"/filtered_netlists/"+results_path.split("/")[-1]
    gds_file = results_path.split("/")[-1].upper() + "_0.gds"
    
    shutil.copy(f"{gds_path}/comp.out", output_path)

    shutil.copy(f"{gds_path}/{gds_file}", output_path)
    os.makedirs(f"{output_path}/primitives", exist_ok=True)

    for gds_elem in os.listdir(f"{gds_path}/2_primitives"):
        if ".python.gds" in gds_elem:
            shutil.copy(f"{gds_path}/2_primitives/{gds_elem}", f"{output_path}/primitives")

def write_hierarchy(results_path, output_path_pre):
    output_path = "/".join(output_path_pre.split("/")[0:-2])
    gds_path = "/".join(results_path.split("/")[0:-2])+"/filtered_netlists/"+results_path.split("/")[-1]
    gds_file = results_path.split("/")[-1].upper() + "_0.gds"

    # Load the original GDS
    lib = gdspy.GdsLibrary(infile=f"{gds_path}/{gds_file}")

    # Get top-level cell (assuming only one top cell)
    top_cell = lib.top_level()[0]

    # New library
    new_lib = gdspy.GdsLibrary()

    cells = list()

    def visit_cells(cell):
        if cell.name not in cells and cell is not top_cell:
            cells.append(cell.name)

        for ref in cell.references:
            visit_cells(ref.ref_cell)


    # Copy top cell with polygon exclusion
    visit_cells(top_cell)

    # Add all copied cells to the new lib
    with open(f"{output_path}/cells.txt", "w") as output_file:
        for cell in cells:
            output_file.write(f"{cell}\n")
        

def main():
    # Set up argparse for command-line arguments
    parser = argparse.ArgumentParser(description="Converts NGSPICE results in CSV files.")
    parser.add_argument("results_path", help="The input NGSPICE simulation")
    parser.add_argument("outputs_path", help="Output csv path")

    # Parse the command-line arguments
    args = parser.parse_args()

    output_path_pre, output_path_post = process_files(args.results_path, args.outputs_path)
    process_csv_files(output_path_pre, output_path_post)
    copy_gds(args.results_path, output_path_pre)
    write_hierarchy(args.results_path, output_path_pre)

# Run the script if it is executed directly
if __name__ == "__main__":
    main()
