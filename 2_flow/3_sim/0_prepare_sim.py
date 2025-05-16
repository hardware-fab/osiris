import argparse
import os
import concurrent.futures
import subprocess
from typing import List


def process_file(netlist_path, gen_nl_path):
    pre_layout_content = None
    post_layout_content = None
    nl_name = netlist_path.split("/")[-1]
    template_file_pre = f"{netlist_path}/{nl_name}_pre.spice"
    template_file_post = f"{netlist_path}/{nl_name}_post.spice"

    curr_itr = 0
    gen_nl = gen_nl_path.split("/")[-1]

    for gen_nl_file in os.listdir(netlist_path + "/filtered_netlists/" + gen_nl):  
        if gen_nl_file == "gds-extracted.spice":
            with open(netlist_path + "/filtered_netlists/" + gen_nl + "/" + gen_nl_file) as file:
                post_layout_content = file.read()
                post_layout_content = post_layout_content.replace("$ **FLOATING", "")
        elif gen_nl_file == f"{gen_nl}.lvs.spice":
            with open(netlist_path + "/filtered_netlists/" + gen_nl + "/" + f"{gen_nl}.lvs.spice") as file:
                pre_layout_content = file.read().split("\n")
                pre_layout_content = [line for line in pre_layout_content if len(line) > 0 and "Created" not in line]
                pre_layout_content = pre_layout_content[1:-1]
                pre_layout_content = "\n".join(pre_layout_content)
                pre_layout_content = pre_layout_content.replace("e-6","")

        if pre_layout_content and post_layout_content:
            os.makedirs(netlist_path + "/simulations/" + gen_nl, exist_ok=True)
            with open(template_file_pre, "r") as tmpl_file_pre:
                template_file_str_pre = tmpl_file_pre.read()
            with open(template_file_post, "r") as tmpl_file_post:
                template_file_str_post = tmpl_file_post.read()
            pre_layout_str = template_file_str_pre.replace("XXX", pre_layout_content)
            post_layout_str = template_file_str_post.replace("XXX", post_layout_content)

            with open(netlist_path + "/simulations/" + gen_nl + "/" + gen_nl + "_pre.spice", "w") as pre_layout_file:
                pre_layout_file.write(pre_layout_str)
            with open(netlist_path + "/simulations/" + gen_nl + "/" + gen_nl + "_post.spice", "w") as post_layout_file:
                post_layout_file.write(post_layout_str)

        a = len(os.listdir(netlist_path + "/filtered_netlists"))
        print(f"Creating simulation context: [{curr_itr}/{a}]", end="\r")
        curr_itr += 1    

def main():
    # Set up argparse for command-line arguments
    parser = argparse.ArgumentParser(description="Simulates the netlists under the given directory with NGSPICE.")
    parser.add_argument("netlists_path", help="The input SPECTRE netlists")
    parser.add_argument("gen_nl", help="The input SPECTRE netlists")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the process_file function with the provided file paths
    process_file(args.netlists_path, args.gen_nl)

# Run the script if it is executed directly
if __name__ == "__main__":
    main()

