import os
import json
import random
import itertools

def load_pairs(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # estraiamo le coppie
    return [tuple(pair) for pair in data.get("pairs", [])]

def build_groups(all_elements, pairs):
    paired = set(x for pair in pairs for x in pair)
    # elementi indipendenti sono quelli non in alcuna coppia
    independents = [e for e in all_elements if e not in paired and 'C' not in e and 'R' not in e]
    # ogni indipendente va trattato come gruppo singolo
    singletons = [(e,) for e in independents]
    return pairs + singletons

def generate_nf_permutations(groups, nf_values):
    """
    Restituisce una lista di dizionari:
      [
        { 'M1':2, 'M2':2, 'M3':4, ... },
        { 'M1':2, 'M2':2, 'M3':6, ... },
        ...
      ]
    """
    all_perms = []
    # per ogni assegnazione di nf_values ai gruppi
    for assignment in itertools.product(nf_values, repeat=len(groups)):
        mapping = {}
        for (group, nf) in zip(groups, assignment):
            for element in group:
                mapping[element] = nf
        all_perms.append(mapping)
    return all_perms

if __name__ == "__main__":
    
    # path del file JSON delle coppie
    circuit_class = "five_transistors_ota" # "miller_ota", "ahuja_ota", "ota_ff"
    pick_N = True # Whether to use all permutations or a subset
    picked_permutations = 10_000
    base_path = f"/path/to/1_input/netlist/{circuit_class}" # Absolute path
    json_path = f"{base_path}/pairs.json"
    input_template_path = f"{base_path}/{circuit_class}.sp"
    output_generation_path = f"{base_path}/unfiltered_netlists/"
    permutations_map_path = f"{base_path}/permutations_map.json"
    
    # valori possibili di NF
    nf_values = {'miller_ota': [2, 4, 6],
                 'ahuja_ota': [2, 4],
                 'ota_ff': [2, 4],
                 'five_transistors_ota': [2, 4, 6, 8, 10, 12, 14, 16]}
    
    # esempio di lista completa di elementi:
    all_elements = []
     # Read the template netlist
    with open(input_template_path, "r") as file:
            template_content = file.read()
    template_content_lines = template_content.split("\n")
    netlist_name = template_content_lines[0].split(" ")[1]
    
    for l in template_content_lines:
        if '.ends' not in l and '.subckt' not in l:
            all_elements.append(l.split(' ')[0])
    
    pairs = load_pairs(json_path)
    
    groups = build_groups(all_elements, pairs)

    permutations = generate_nf_permutations(groups, nf_values[circuit_class])
    if pick_N:
        N = min(picked_permutations, len(permutations))
        permutations = random.sample(permutations, N)
        
    permutations_map = {}
    
    if not os.path.exists(output_generation_path):
        os.makedirs(output_generation_path)
    
    for idx, permutation in enumerate(permutations):
        print(f"[I] Permutation {permutation} for netlist {netlist_name}_{idx}")
        
        new_lines = []     
        
        for line_id, line in enumerate(template_content_lines):
            if line_id == 0:
                new_lines.append(template_content_lines[0].replace(netlist_name, f'{netlist_name}_{idx}'))
            elif line_id == len(template_content_lines) - 1:
                new_lines.append(template_content_lines[-1].replace(netlist_name, f'{netlist_name}_{idx}'))
            elif 'M' in line.split(" ")[0]:
                for device in permutation:
                    if device == line.split(' ')[0]:
                        new_lines.append(line.replace('nf=ZZZ', f'nf={permutation[device]}'))
            else:
                new_lines.append(line)
        
        output_netlist_path = os.path.join(output_generation_path, f"{netlist_name}_{idx}", f"{netlist_name}_{idx}.sp")
        otuput_dir_path = os.path.join(output_generation_path, f"{netlist_name}_{idx}")
        os.makedirs(otuput_dir_path, exist_ok=True)
        with open(output_netlist_path, "w") as output_file:
            for new_line in new_lines:
                output_file.write(new_line + '\n')
        
        permutations_map[f'{netlist_name}_{idx}'] = permutation
    
    with open(permutations_map_path, 'w') as json_file:
        json.dump(permutations_map, json_file, indent=4)