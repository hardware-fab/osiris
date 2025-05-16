import argparse
import gdspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def rmse(prediction, target):
    return np.sqrt(((prediction - target) ** 2).mean())

def pex_score(pre_layout, post_layout, tolerance=0.1, weight=0):
    n_samples = len(pre_layout)
    out_of_bounds_indexes = list()

    for sample_itr in range(0, n_samples):
        lower_bound = pre_layout[sample_itr] * (1-tolerance) 
        upper_bound = pre_layout[sample_itr] * (1+tolerance)

        if not (lower_bound <= post_layout[sample_itr] <= upper_bound):
            out_of_bounds_indexes.append(sample_itr)

    layout_rmse = rmse(pre_layout, post_layout)
    oob_weight = weight * len(out_of_bounds_indexes)

    return np.round(layout_rmse + oob_weight, decimals=15)

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
        i=0
        for score_values in score:
            if i != len(score)-1:
                plt.axvline(x=score_values[1], color='black', linestyle='--', linewidth=1)
            i+=1
        plt.xlabel('Frequency')
        plt.ylabel('Output')
        plt.ylim(-2, 2)
        plt.title(f'PEX Score: low: {score[0][2]:.4f}, mid: {score[1][2]:.4f}, high: {score[2][2]:.4f}')
        plt.legend()
        pdf.savefig()
        plt.close()

def save_to_npy(output_file, score):
    np.save(output_file, np.array(score))

def process_csv_files(output_path, mode='rl'):
    if mode == 'rl':
        variant_name = output_path.split("/")[-1].split('__')[0]
    else:
        variant_name = output_path.split("/")[-3]
    
    output_path_pre = f"{output_path}/{variant_name}_pre.csv"
    output_path_post = f"{output_path}/{variant_name}_post.csv"

    pre_frequency, pre_out = read_csv_file(output_path_pre)
    post_frequency, post_out = read_csv_file(output_path_post)

    frequency_range = int(len(pre_frequency) / 3)
    frequencies = []

    for i in range(3):
        start = i * frequency_range
        end = (i + 1) * frequency_range if i < 2 else len(pre_frequency)
        frequencies.append((start, end))

    # Compute scores for each frequency range
    score_low_f  = pex_score(pre_out[frequencies[0][0]:frequencies[0][1]], post_out[frequencies[0][0]:frequencies[0][1]])
    score_mid_f  = pex_score(pre_out[frequencies[1][0]:frequencies[1][1]], post_out[frequencies[1][0]:frequencies[1][1]])
    score_high_f = pex_score(pre_out[frequencies[2][0]:frequencies[2][1]], post_out[frequencies[2][0]:frequencies[2][1]])

    # Prepare scores as a structured array
    scores = np.array([
        [frequencies[0][0], frequencies[0][1], score_low_f],
        [frequencies[1][0], frequencies[1][1], score_mid_f],
        [frequencies[2][0], frequencies[2][1], score_high_f]
    ])

    plot_to_pdf((pre_frequency, pre_out), (post_frequency, post_out), scores, f"{output_path}/score.pdf")
    save_to_npy(f"{output_path}/score.npy", scores)

def save_area(output_path, mode='rl'):
    if mode == 'rl':
        gds_file = output_path.split("/")[-1].upper().split('__')[0] + "_0.gds"
    else:
        gds_file = output_path.split("/")[-3].upper() + "_0.gds"

   # Load the GDSII file
    lib = gdspy.GdsLibrary(infile=f"{output_path}/{gds_file}")

    # Get the top-level cells
    top_cells = lib.top_level()
    if not top_cells:
        raise ValueError("No top-level cells found in GDS file.")

    total_area = 0.0

    for cell in top_cells:
        # Iterate over all polygons in the cell
        for ref in cell.references:
                total_area += (ref.get_bounding_box()[1][0]-ref.get_bounding_box()[0][0])*\
                              (ref.get_bounding_box()[1][1]-ref.get_bounding_box()[0][1])
    
    total_area_magnitude = np.floor(np.log10(total_area))
    unit_magnitude = np.floor(np.log10(lib.unit))
    
    save_to_npy(f"{output_path}/area.npy", total_area/pow(10, total_area_magnitude - unit_magnitude))
    
def get_pex_score(output_path, mode='rl'):
    process_csv_files(output_path, mode)
    save_area(output_path, mode)

def main():
    # Set up argparse for command-line arguments
    parser = argparse.ArgumentParser(description="Converts NGSPICE results in CSV files.")
    parser.add_argument("output_path", help="Output csv path")

    # Parse the command-line arguments
    args = parser.parse_args()

    process_csv_files(args.output_path)
    save_area(args.output_path)

# Run the script if it is executed directly
if __name__ == "__main__":
    main()