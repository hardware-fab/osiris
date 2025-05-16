##############################################
## VERSION 2 #################################
##############################################
import os
import sys
import glob
import time
import json
import math
import torch
import gdspy
import shutil
import random
import subprocess
import portalocker
import setproctitle
import multiprocessing
import concurrent.futures

import numpy as np
import torch.nn as nn
import torch.optim as optim
   
from copy import deepcopy
from pex import get_pex_score

random.seed(123)

CIRCUIT_CLASS = 'five_transistors_ota'

EXPLORATION_MODE = 'rl' # 'dataset' or 'rl'
MULTI_PROCS = True

NUM_VARIANTS = {'miller_ota': 128,
                'ahuja_ota': 128,
                'five_transistors_ota': 128,
                'ota_ff': 128}
WD = f"/path/to/working/dir"
BASE_PATH = f"{WD}/1_input/netlist/{CIRCUIT_CLASS}"
WORKING_PATH = f"{BASE_PATH}/filtered_netlists"
UNFILTERED_PATH = f"{BASE_PATH}/unfiltered_netlists"
PERMUTATIONS_MAP_PATH = f"{BASE_PATH}/permutations_map.json"
OUTPUT_DIR = {'dataset': '3_output',
              'rl': '3_output_rl'}

MAX_PROCS = 10
MAX_RUNS = {'five_transistors_ota': 10,
            'miller_ota': 10,
            'ahuja_ota': 10,
            'ota_ff': 10}
MAX_RND_MOVES = 100

# --------------------------
# Configuration and Settings
# TD stands for the agent in charge of deciding which 'Tile' to move along which 'Direction'.
# F stands for the agente in charge of deciding which 'Fingers' configurations to explore.
# --------------------------
NUM_TILES = {'five_transistors_ota': 5,
             'miller_ota': 11,
             'ahuja_ota': 15,
             'ota_ff': 13}
NUM_DEVICES = {'five_transistors_ota': 5,
               'miller_ota': 9,
               'ahuja_ota': 14,
               'ota_ff': 13} 

NUM_FEATURES = 8 # [x_ur, y_ur, x_ll, y_ll, one-hot-enc-4bits]
STATE_TD_DIM = NUM_FEATURES * NUM_TILES[CIRCUIT_CLASS]  # each tile is defined by 4 numbers
STATE_F_DIM = NUM_DEVICES[CIRCUIT_CLASS] # one input element for each device, it represents the number of fingers ot that device
NUM_DIRECTIONS = 4         # 0: right, 1: left, 2: up, 3: down
HIDDEN_TD_UNITS = NUM_TILES[CIRCUIT_CLASS] * 128
NUM_TD_LAYERS = 5
NUM_F_LAYERS = 5
HIDDEN_F_UNITS = NUM_DEVICES[CIRCUIT_CLASS] * 64
POSSIBLE_FINGERS = {'miller_ota': [2, 4, 6],
                    'ahuja_ota': [2, 4],
                    'ota_ff': [2, 4],
                    'five_transistors_ota': [2, 4, 6, 8, 10, 12, 14, 16]}   # It reports the possible values assumed by 'number of fingers' of each device

# Hyperparameters
EPISODES_TD = {'five_transistors_ota': 100,
               'miller_ota': 100,
               'ahuja_ota': 100,
               'ota_ff': 100} # Ahuja could be pushed up to 300, since it has a slower learning rate.
EPISODES_F = {'five_transistors_ota': 128,
              'miller_ota': 128,
              'ahuja_ota': 128,
              'ota_ff': 128}
GAMMA = 0.99
LR_TD = 1e-5
LR_F = 1e-5
REWARD_PENALTY_TD = -0.001
REWARD_PENALTY_F = -10
ENTROPY_COEF = 0.01

MAX_STEPS = 30  
ROLLOUT_SIZE = 128
MINIBATCH_SIZE = 16

ALPHA = 5
BETA = 1.5

MOVE = 100 # Amount of nanometers to move each tile each time a direction is chosen

TILE_TYPES = ['N', 'P', 'C', 'R']
TILE_TYPE_TO_ONEHOT = {
    'N': [1, 0, 0, 0], # NMOS
    'P': [0, 1, 0, 0], # PMOS
    'C': [0, 0, 1, 0], # CAPACITOR
    'R': [0, 0, 0, 1], # RESISTOR
}

# Folder-level lock
class FolderLock:
    def __init__(self, folder_path):
        self.folder_path = os.path.abspath(folder_path)
        self.lock_file_path = os.path.join(self.folder_path, '.folder.lock')
        self.lock_file = None

    def __enter__(self):
        os.makedirs(self.folder_path, exist_ok=True)
        # Open the lock file for writing
        self.lock_file = open(self.lock_file_path, 'w')
        print(f"Waiting for lock on {self.folder_path}...")
        portalocker.lock(self.lock_file, portalocker.LOCK_EX)  # Blocking lock
        print(f"Lock acquired on {self.folder_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        portalocker.unlock(self.lock_file)
        self.lock_file.close()
        print(f"Lock released on {self.folder_path}")

def worker(permutation_queue, success_counter, success_limit, counter_lock):
    while True:
        try:
            permutation = permutation_queue.get_nowait()
        except:
            break  # Queue empty

        baseline_created = create_place_route_baseline(permutation)
        if baseline_created:
            result = variants_creation(permutation)
            with counter_lock:
                if success_counter.value < success_limit:
                    success_counter.value += 1
                    print(f"Success #{success_counter.value} for permutation {permutation}")
                    return (permutation, result)
                else:
                    print(f"Ignoring result for permutation {permutation} (limit reached)")
                    return None  # Do not count extra successes
        else:
            print(f"Failed to create baseline for {permutation}")
            continue

# --------------------------
# Neural Networks: Actor-Critic (PPO)
# --------------------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim, num_tiles, num_directions, num_layers=2, hidden_units=128):
        super(ActorCritic, self).__init__()
        
        # Create shared layers with the given number of layers and units
        layers = []
        input_dim = state_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_units))
            layers.append(nn.LayerNorm(hidden_units))
            layers.append(nn.ReLU())
            input_dim = hidden_units
        
        self.actor = nn.Sequential(*layers)
        self.critic = nn.Sequential(*layers)
        
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        for layer in self.critic:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # Actor (policy network)
        self.actor_tile_head = nn.Linear(hidden_units, num_tiles)  # tile selection
        self.actor_dir_head = nn.Linear(hidden_units, num_directions)  # direction selection
        
        # Critic (value network)
        self.critic_head = nn.Linear(hidden_units, 1)  # value of the state
    
    def forward(self, x):
        x_actor = self.actor(x)
        x_critic = self.critic(x)
        tile_probs = torch.softmax(self.actor_tile_head(x_actor), dim=-1)  # action probabilities
        dir_probs = torch.softmax(self.actor_dir_head(x_actor), dim=-1)
        state_value = self.critic_head(x_critic)  # value of the state
        return tile_probs, dir_probs, state_value
#MetaAgent(STATE_F_DIM, NUM_DEVICES[CIRCUIT_CLASS], len(POSSIBLE_FINGERS[CIRCUIT_CLASS]), num_layers=NUM_F_LAYERS, hidden_units=HIDDEN_F_UNITS)

class MetaAgent(nn.Module):
    def __init__(self, state_dim, num_devices, fingers, num_layers=2, hidden_units=128):
        super(MetaAgent, self).__init__()
        self.num_devices = num_devices
        self.fingers = fingers
        
        # Create shared layers with the given number of layers and units
        layers = []
        input_dim = state_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_units))
            layers.append(nn.LayerNorm(hidden_units))
            layers.append(nn.ReLU())
            input_dim = hidden_units
        
        self.actor = nn.Sequential(*layers)
        
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # Actor (policy network)
        self.actor_head = nn.Linear(hidden_units, self.num_devices * self.fingers)  # tile selection
    
    def forward(self, x):
        x_actor = self.actor(x)
        logits = self.actor_head(x_actor)  # action probabilities
        logits = logits.view(-1, self.num_devices, self.fingers)
        fingers_probs = torch.softmax(logits, dim=-1)
        return fingers_probs

# --------------------------
# Action Selection (PPO)
# --------------------------
def select_action(state, policy_net, device):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    tile_probs, dir_probs, _ = policy_net(state_tensor)

    tile_dist = torch.distributions.Categorical(tile_probs)
    dir_dist = torch.distributions.Categorical(dir_probs)

    tile_action = tile_dist.sample()
    dir_action = dir_dist.sample()

    return tile_action.item(), dir_action.item(), tile_dist.log_prob(tile_action).item(), dir_dist.log_prob(dir_action).item()

# --------------------------
# PPO Loss Function
# --------------------------
def compute_ppo_loss(old_tile_log_probs, old_dir_log_probs,
                     new_tile_log_probs, new_dir_log_probs,
                     advantages, new_values, returns,
                     epsilon=0.2):
    # PPO ratio using log probs
    ratio_tile = (new_tile_log_probs - old_tile_log_probs).exp()
    ratio_dir = (new_dir_log_probs - old_dir_log_probs).exp()

    # Clipped objective
    clipped_ratio_tile = ratio_tile.clamp(1 - epsilon, 1 + epsilon)
    clipped_ratio_dir = ratio_dir.clamp(1 - epsilon, 1 + epsilon)

    surrogate_tile = ratio_tile * advantages
    surrogate_dir = ratio_dir * advantages
    clipped_surrogate_tile = clipped_ratio_tile * advantages
    clipped_surrogate_dir = clipped_ratio_dir * advantages

    # Value function loss
    value_loss = (new_values.squeeze() - returns).pow(2)

    # Total loss
    loss = -torch.min(surrogate_tile, clipped_surrogate_tile).mean() \
           -torch.min(surrogate_dir, clipped_surrogate_dir).mean() \
           + 0.5 * value_loss.mean()

    return loss

# --------------------------
# Training Function (PPO)
# --------------------------
def train(device, policy_net, optimizer, rollout, chosen_sample):
    if len(rollout) < ROLLOUT_SIZE:
        return False
    
    # Sample a batch of experiences
    states, actions, old_tile_log_probs, old_dir_log_probs, rewards, next_states, dones = zip(*rollout)
    
    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor(np.array(actions)).to(device)
    old_tile_log_probs = torch.FloatTensor(np.array(old_tile_log_probs)).to(device)
    old_dir_log_probs = torch.FloatTensor(np.array(old_dir_log_probs)).to(device)
    rewards = torch.FloatTensor(np.array(rewards)).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones = torch.FloatTensor(np.array(dones)).to(device)
    
    dataset = torch.utils.data.TensorDataset(
        states, actions, old_tile_log_probs, old_dir_log_probs, rewards, next_states, dones
    )
    
    ppo_epochs = max(1, len(rollout) // MINIBATCH_SIZE)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=MINIBATCH_SIZE, shuffle=True, drop_last=True)
    
    for ppo_epoch in range(ppo_epochs):
        for batch in dataloader:
            batch_states, batch_actions, batch_old_tile_log_probs, batch_old_dir_log_probs, batch_rewards, batch_next_states, batch_dones = batch
    
            # Compute advantages and value targets (Generalized Advantage Estimation)
            _, _, state_values = policy_net(batch_states.flatten(start_dim=1))
            _, _, next_state_values = policy_net(batch_next_states.flatten(start_dim=1))
            
            deltas = batch_rewards + GAMMA * next_state_values.squeeze() * (1 - batch_dones) - state_values.squeeze()
            advantages = deltas.detach()
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
            
            # Compute new probabilities
            tile_probs, dir_probs, new_state_values = policy_net(batch_states.flatten(start_dim=1))
            
            # Gather new log probabilities
            new_tile_log_probs = tile_probs.gather(1, batch_actions[:, 0].unsqueeze(1)).squeeze(1).log()
            new_dir_log_probs = dir_probs.gather(1, batch_actions[:, 1].unsqueeze(1)).squeeze(1).log()
            
            # Compute returns
            returns = batch_rewards + GAMMA * next_state_values.squeeze() * (1 - batch_dones)
            returns = returns.detach()
            
            # Compute loss
            loss = compute_ppo_loss(batch_old_tile_log_probs, batch_old_dir_log_probs, new_tile_log_probs, new_dir_log_probs, advantages, new_state_values, returns)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"\tSample: {chosen_sample} | PPO update epoch {ppo_epoch}/{ppo_epochs}")
    
    return True

# --------------------------
scripts_reroute = [
            "1_reroute_sch2lay.sh",
            "2_reroute_lvs.sh",
            "3_reroute_sim.sh",
            "4_reroute_postprocess.sh"
          ]

def generate_input_state(all_elements_info, min_coord, max_coord):
    
    input_state = []
    
    for cell in all_elements_info:
        bbox = (cell['bbox'].flatten() - min_coord) / (max_coord - min_coord)
        type = np.array(TILE_TYPE_TO_ONEHOT[cell['type']])
        cell_data = np.concatenate((bbox, type))
        input_state.append(cell_data)
    
    return np.array(input_state).flatten()

def load_dataset(num_samples, configs_path):
    dataset = []
    circuits = sorted(os.listdir(configs_path), key=lambda d: os.stat(os.path.join(configs_path, d)).st_ctime)

    num_samples = min(num_samples, len(circuits))
    
    for sample in range(num_samples):
        dataset.append(os.path.join(configs_path, circuits[sample]))

    return dataset

def process_cell(cell):
    
    all_elements_info = []
    
    for ref in cell.references:
        if len(ref.ref_cell.references) > 0:
            for rref in ref.ref_cell.references:
                bbox = rref.get_bounding_box()
                bbox[bbox < 1e-9] = 0.0
                
                cell_name = rref.ref_cell.name
                
                if 'PMOS' in cell_name:
                    cell_type = 'P'
                elif 'NMOS' in cell_name:
                    cell_type = 'N'
                elif 'CAP' in cell_name:
                    cell_type = 'C'
                elif 'RES' in cell_name:
                    cell_type = 'R'
                else:
                    sys.exit(f"[E] No valid type for cell {cell_name}.")
                
                for el in all_elements_info:
                    if cell_name == el['cell']:
                        cell_name = cell_name + "_1"
                
                all_elements_info.append(({'cell': cell_name,
                                        'type': cell_type,
                                        'bbox': bbox}))
        else:
            bbox = ref.get_bounding_box()
            bbox[bbox < 1e-9] = 0.0
            
            cell_name = ref.ref_cell.name
            
            if 'PMOS' in cell_name:
                cell_type = 'P'
            elif 'NMOS' in cell_name:
                cell_type = 'N'
            elif 'CAP' in cell_name:
                cell_type = 'C'
            elif 'RES' in cell_name:
                cell_type = 'R'
            else:
                sys.exit(f"[E] No valid type for cell {cell_name}.")
            
            for el in all_elements_info:
                if cell_name == el['cell']:
                    cell_name = cell_name + "_1"
            
            all_elements_info.append(({'cell': cell_name,
                                    'type': cell_type,
                                    'bbox': bbox}))

    return all_elements_info

def get_minmax_coords(all_elements_info):
    
    min_coord = None
    max_coord = None
    
    for item in all_elements_info:
        if min_coord == None or min_coord > min(item['bbox'].flatten()):
            min_coord = min(item['bbox'].flatten())
        
        if max_coord == None or max_coord < max(item['bbox'].flatten()):
            max_coord = max(item['bbox'].flatten())
    
    return min_coord, max_coord
 
def reset_environment(episode, chosen_sample):
    
    gds_file = os.path.join(chosen_sample, chosen_sample.split("/")[-1].upper() + "_0.gds")
    layout = gdspy.GdsLibrary(infile=gds_file)
    top_cell = layout.top_level()[0]

    all_elements_info = process_cell(top_cell)

    # Display result
    if episode == 0:
        for item in all_elements_info:
            print(f"Cell: {item['cell']}, Type: {item['type']}, BBox: {np.array(item['bbox']).flatten()}")
    
    min_coord, max_coord = get_minmax_coords(all_elements_info)
    
    input_state = generate_input_state(all_elements_info, min_coord, max_coord)

    return input_state

def step_environment(tile_action, direction_action, episode, chosen_sample, num_steps, accumulated_transforms, 
                     bl_metrics, run_path, f_episode):
    """
    Given the current state and a composite action (tile and direction),
    returns next_state, reward, done.
    
    Note: You must implement the logic to update the state,
    ensuring constraints (e.g., no overlapping) are enforced.
    For now, this is a dummy implementation.
    """    
    working_sample = os.path.join(WORKING_PATH, chosen_sample.split("/")[-1])
    
    src = os.path.join(working_sample, working_sample.split('/')[-1].upper()) + '_0.gds'
    dst = src + '.bckp'
    result = subprocess.run(["cp", src, dst], check=True)
    if result.returncode != 0:
        print("GDS copy failed. Exiting.")
        sys.exit(1)
    
    # 0: right, 1: left, 2: up, 3: down
    if direction_action == 0: # right
        accumulated_transforms[tile_action] += np.array([-MOVE, 0])
    elif direction_action == 1: # left
        accumulated_transforms[tile_action] += np.array([MOVE, 0])
    elif direction_action == 2: # up
        accumulated_transforms[tile_action] += np.array([0, MOVE])
    else: # down
        accumulated_transforms[tile_action] += np.array([0, -MOVE])
    
    str_acc_transforms = ''
    for t in accumulated_transforms:
        str_acc_transforms += f"{t[0]},{t[1]};"
    
    # Call to HORUS
    for i, script in enumerate(scripts_reroute, start=1):
        script_path = f"/work/gchiari/ADG_PERSONAL_mpiccoli/2_flow/5_reroute/{script}"
        result = subprocess.run(
            ['bash', script_path, str(working_sample), str(tile_action), str(direction_action), str(episode), str(str_acc_transforms), str(run_path), str(f_episode), str('rl')],
            capture_output=True,
            text=True
        )
        #print(f"Return code after {script} is: {result.returncode}")
        if result.returncode != 0:
            #print(f"❌ Script {script} failed. Stopping execution.")
            subprocess.run(["cp", dst, src], check=True)
            subprocess.run(["rm", dst], check=True)
            break
    
    if result.returncode == 0:
        sample = chosen_sample.split("/")[-1]
        variant_folder = os.path.join("/work/gchiari/ADG_PERSONAL_mpiccoli/", run_path, f"iter_{f_episode}_{sample}", f"{sample}_{episode}")
        reward, reward_pex, reward_area, done = get_reward(num_steps, bl_metrics, variant_folder) # compute reward
        np.save(os.path.join(variant_folder, "applied_transforms.npy"), np.array(accumulated_transforms))
    else:
        reward = REWARD_PENALTY_TD # negative fixed penalty if the action taken makes the flow crash.
        reward_pex = 0
        reward_area = 0
        if direction_action == 0: # right
            accumulated_transforms[tile_action] -= np.array([-MOVE, 0])
        elif direction_action == 1: # left
            accumulated_transforms[tile_action] -= np.array([MOVE, 0])
        elif direction_action == 2: # up
            accumulated_transforms[tile_action] -= np.array([0, MOVE])
        else: # down
            accumulated_transforms[tile_action] -= np.array([0, -MOVE])
        done = True
    
    return reward, reward_pex, reward_area, done, accumulated_transforms

def get_reward(num_steps, bl_metrics, variant_folder):
    
    get_pex_score(variant_folder)
    
    pex_score = np.mean(np.load(os.path.join(variant_folder, "score.npy"))[:, -1])
    area = float(np.load(os.path.join(variant_folder, "area.npy")))
    
    if area < bl_metrics[0]:
        reward_area = np.abs(area - bl_metrics[0]) * 1e08 * BETA
    else:
        reward_area = np.abs(area - bl_metrics[0]) * 1e08 * (-BETA)
    
    if pex_score < bl_metrics[1]:
        reward_pex = np.abs(pex_score - bl_metrics[1]) * ALPHA
    else:
         reward_pex = np.abs(pex_score - bl_metrics[1]) * (-ALPHA)
    
    reward = reward_pex + reward_area

    done = False
    if num_steps >= MAX_STEPS:
        done = True

    return reward, reward_pex, reward_area, done

def get_next_state(chosen_sample):
    gds_file = os.path.join(chosen_sample, chosen_sample.split("/")[-1].upper() + "_0.gds")
    layout = gdspy.GdsLibrary(infile=gds_file)
    top_cell = layout.top_level()[0]

    all_elements_info = process_cell(top_cell)
    
    min_coord, max_coord = get_minmax_coords(all_elements_info)
        
    next_state = generate_input_state(all_elements_info, min_coord, max_coord)
    
    return next_state

def load_bl_metrics(chosen_sample):
    
    bl_path = os.path.join(f"{WD}/{OUTPUT_DIR[EXPLORATION_MODE]}", chosen_sample.split("/")[-1])
    
    get_pex_score(bl_path)
    
    pex_score = np.mean(np.load(os.path.join(bl_path, "score.npy"))[:, -1])
    area = float(np.load(os.path.join(bl_path, "area.npy")))
    
    return area, pex_score

def get_state_permutation(chosen_id, permutations, device):
    permutation = permutations[chosen_id]
    state_f = []
    
    for d in permutation:
        state_f.append(permutation[d])
    
    return torch.FloatTensor(state_f).unsqueeze(0).to(device)

def get_netlist_name(selected_permutation, permutations):
    
    num_fingers = [POSSIBLE_FINGERS[CIRCUIT_CLASS][nf] for nf in selected_permutation]
    
    matched_key = None
    
    for key, subdict in permutations.items():
        # Sort subdict by key to ensure consistent order
        values = [v for k, v in sorted(subdict.items())]
        if values == num_fingers:
            matched_key = key
    
    return matched_key

def create_place_route_baseline(chosen_sample):
        
    subdir = os.path.join(UNFILTERED_PATH, chosen_sample)
    
    src_path = os.path.join(BASE_PATH, f"{CIRCUIT_CLASS}.const.json")
    dst_path = os.path.join(subdir, f"{subdir.split('/')[-1]}.const.json")
    shutil.copy(src_path, dst_path)
    
    pdk = f"{WD}/0_install/pdk/SKY130_PDK"
    task_pr = f"schematic2layout"
    result = subprocess.run(
        [task_pr, str(subdir), '-p', str(pdk), '-w', str(subdir)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return False
    
    task_lvs = f"{WD}/2_flow/1_lvs/lvs.sh"
    result = subprocess.run(
        ['bash', task_lvs, str(subdir)],
        capture_output=True,
        text=True
    )
    if "Top level cell failed pin matching" in result.stdout or "Netlists do not match" in result.stdout:
        return False
    
    task_validate = f"{WD}/2_flow/2_validate/validate.py"
    result = subprocess.run(
        ['python', task_validate, str(BASE_PATH), str(subdir)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return False
    
    task_sim = f"{WD}/2_flow/3_sim/sim.sh"
    result = subprocess.run(
        ['bash', task_sim, str(BASE_PATH), str(subdir)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return False
    
    task_postprocess = f"{WD}/2_flow/4_postprocess/postprocess.sh"
    result = subprocess.run(
        ['bash', task_postprocess, str(BASE_PATH), str(subdir), str(OUTPUT_DIR[EXPLORATION_MODE])],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return False
    
    return True

# --------------------------
# Train loop function - over tiles and directions
# --------------------------
def train_loop(chosen_sample, run_path, f_episode, policy_net, optimizer, device):
    
    train_ep_start_time = time.time()
    bl_metrics_path = f"{WD}/{OUTPUT_DIR[EXPLORATION_MODE]}/{chosen_sample.split('/')[-1]}"
    
    bl_area, bl_pex_score = load_bl_metrics(bl_metrics_path)
    
    reset_rollout = True
    rewards = []
    
    for ep in range(EPISODES_TD[CIRCUIT_CLASS]):
        start_state = reset_environment(ep, chosen_sample)
        accumulated_transforms = np.array([np.array([0,0]) for _ in range(NUM_TILES[CIRCUIT_CLASS])])
        if reset_rollout:
            rollout = []
        total_reward = 0
        total_r_pex = 0
        total_r_area = 0
        done = False
        num_steps = 0
        state = deepcopy(start_state)
        while not done:
            tile_action, direction_action, old_tile_log_prob, old_dir_log_prob = select_action(state, policy_net, device)
            id = f"_ep{ep}_step{num_steps}"
            reward, reward_pex, reward_area, done, accumulated_transforms = step_environment(tile_action, direction_action, id, chosen_sample, 
                                                                                            num_steps, accumulated_transforms, [bl_area, bl_pex_score], 
                                                                                            run_path, f_episode)
            next_state = get_next_state(chosen_sample)
            rollout.append((state, (tile_action, direction_action), old_tile_log_prob, old_dir_log_prob, reward, next_state, float(done)))
            state = next_state
            total_reward += reward
            total_r_pex += reward_pex
            total_r_area += reward_area
            num_steps += 1
        
        rewards.append(total_reward)

        # Train after each episode
        reset_rollout = train(device, policy_net, optimizer, rollout, chosen_sample)
        
        ep_dir = os.path.join(WD, run_path, f"iter_{f_episode}_{chosen_sample.split('/')[-1]}", f"{chosen_sample.split('/')[-1]}_{ep}")
        if os.path.isdir(ep_dir):
            np.save(os.path.join(ep_dir, "episode_exec_time.npy"),
                [ (time.time() - train_ep_start_time) / 60 ])
        
        print(f"\tRun {run_path.split('/')[-1]} | Iter {f_episode} | Sample: {chosen_sample.split('/')[-1]} | Episode {ep} | Steps: {num_steps:.2f} | Reward: {total_reward:.4f} \
        Reward PEX: {total_r_pex:.6f} | Reward Area: {total_r_area:.6f}")

    return max(rewards)

def train_loop_metagent(permutation, permutations, run):
    
    metagent_start_time = time.time()
    
    setproctitle.setproctitle(f"run_{run}_{permutation}")
    
    # --------------------------
    # Initialize Networks and Optimizer
    # --------------------------
    # Meta agent
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fingers_net = MetaAgent(STATE_F_DIM, NUM_DEVICES[CIRCUIT_CLASS], len(POSSIBLE_FINGERS[CIRCUIT_CLASS]), num_layers=NUM_F_LAYERS, hidden_units=HIDDEN_F_UNITS).to(device)
    optimizer_fingers = optim.Adam(fingers_net.parameters(), lr=LR_F)
    
    # TD agent
    policy_net = ActorCritic(STATE_TD_DIM, NUM_TILES[CIRCUIT_CLASS], NUM_DIRECTIONS, num_layers=NUM_TD_LAYERS, hidden_units=HIDDEN_TD_UNITS).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=LR_TD)
    
    explored_samples = []
    
    run_path = os.path.join("./", OUTPUT_DIR[EXPLORATION_MODE], f"{CIRCUIT_CLASS}_run_{run}")
    os.makedirs(run_path, exist_ok=True)
    
    for f_episode in range(EPISODES_F[CIRCUIT_CLASS]):
        iter_start_time = time.time()
        
        valid_chosen_sample = False
        chosen_sample = None
        
        state_f = get_state_permutation(permutation, permutations, device)
        permutation_probs = fingers_net(state_f)
        permutation_dist = torch.distributions.Categorical(permutation_probs)
        selected_permutation = permutation_dist.sample()[0]
        
        while not valid_chosen_sample:            
            chosen_sample = get_netlist_name(selected_permutation, permutations)
            
            if chosen_sample and (chosen_sample not in explored_samples):
                valid_chosen_sample = True
            else:
                valid_chosen_sample = False
            
            if not valid_chosen_sample:
                permutation_probs = fingers_net(state_f)
                permutation_dist = torch.distributions.Categorical(permutation_probs)
                selected_permutation = permutation_dist.sample()[0]
        
        working_sample = f"{WORKING_PATH}/{chosen_sample}"
        with FolderLock(working_sample):
            complete = create_place_route_baseline(chosen_sample)
            if complete:
                print(f"Working with {chosen_sample}")
                reward = train_loop(working_sample, run_path, f_episode, policy_net, optimizer, device)
                np.save(os.path.join(WD, run_path, f"iter_{f_episode}_{chosen_sample}", "iter_exec_time.npy"),
                        [ (time.time() - iter_start_time) / 60 ])
            else:
                print(f"Failed to create baseline of {working_sample} in run {run}")
                reward = REWARD_PENALTY_F
        permutation = chosen_sample
        explored_samples.append(chosen_sample)

        permutation_log_prob = permutation_dist.log_prob(selected_permutation)  # shape: [num_devices]
        permutation_log_prob_sum = permutation_log_prob.sum()             # sum over devices if needed
        loss = -permutation_log_prob_sum * reward
        
        optimizer_fingers.zero_grad()
        loss.backward()
        optimizer_fingers.step()
        
        print(f"Run: {run} | Iter {f_episode} | Reward: {reward:.4f}")
    
    np.save(os.path.join(WD, run_path, "run_exec_time.npy"),
            [ (time.time() - metagent_start_time) / 60 ])

# --------------------------
# Dataset creation mode
# --------------------------
def variants_creation(selected_permutation):
    
    print(f"Working with {selected_permutation}")
    
    accumulated_transforms = np.array([np.array([0,0]) for _ in range(NUM_TILES[CIRCUIT_CLASS])])
    
    working_sample = os.path.join(WORKING_PATH, selected_permutation)
    
    output_path = os.path.join("./", OUTPUT_DIR[EXPLORATION_MODE])
    
    permutation_start_time = time.time()
    
    for id_mov in range(MAX_RND_MOVES):
        iter_start_time = time.time()
        
        valid_flag = False
        rnd_tile = -1
        rnd_direction = -1
        
        while not valid_flag:
            # Random choice of tile and direction
            rnd_tile = random.randint(0, NUM_TILES[CIRCUIT_CLASS]-1)
            rnd_direction = random.randint(0, NUM_DIRECTIONS-1)
            
            src = os.path.join(working_sample, working_sample.split('/')[-1].upper()) + '_0.gds'
            dst = src + '.bckp'
            shutil.copy(src, dst)
            
            if rnd_direction == 0: # right
                accumulated_transforms[rnd_tile] += np.array([-MOVE, 0])
            elif rnd_direction == 1: # left
                accumulated_transforms[rnd_tile] += np.array([MOVE, 0])
            elif rnd_direction == 2: # up
                accumulated_transforms[rnd_tile] += np.array([0, MOVE])
            else: # down
                accumulated_transforms[rnd_tile] += np.array([0, -MOVE])
            
            str_acc_transforms = ''
            for t in accumulated_transforms:
                str_acc_transforms += f"{t[0]},{t[1]};"
            
            # Call to HORUS
            for i, script in enumerate(scripts_reroute, start=1):
                script_path = f"{WD}/2_flow/5_reroute/{script}"
                result = subprocess.run(
                    ['bash', script_path, str(working_sample), str(rnd_tile), str(rnd_direction), str(''), str(str_acc_transforms), str(output_path), str(id_mov), str('dataset')],
                    capture_output=True,
                    text=True
                )
                #print(f"Return code after {script} is: {result.returncode}")
                if result.returncode != 0:
                    if os.path.isfile(dst):
                        shutil.copy(dst, src)
                        os.remove(dst)
                    valid_flag = False
                    if rnd_direction == 0: # right
                        accumulated_transforms[rnd_tile] -= np.array([-MOVE, 0])
                    elif rnd_direction == 1: # left
                        accumulated_transforms[rnd_tile] -= np.array([MOVE, 0])
                    elif rnd_direction == 2: # up
                        accumulated_transforms[rnd_tile] -= np.array([0, MOVE])
                    else: # down
                        accumulated_transforms[rnd_tile] -= np.array([0, -MOVE])
                    break
                else:
                    valid_flag = True
        
        if valid_flag:
            variant_folder = os.path.join(WD, output_path, f"{selected_permutation}", "variants", f"iter_{id_mov}")
            get_pex_score(variant_folder, mode='dataset')           

            np.save(os.path.join(variant_folder, "variant_creation_time.npy"), [ (time.time() - iter_start_time) / 60 ])
            np.save(os.path.join(variant_folder, "applied_transforms.npy"), np.array(accumulated_transforms))
            
            print(f"Done iteration {id_mov} / {MAX_RND_MOVES} for {selected_permutation}")
            
    np.save(os.path.join(WD, output_path, f"{selected_permutation}", "variants_creation_time.npy"),
            [ (time.time() - permutation_start_time) / 60 ])

# --------------------------
# Training Loop - over fingers
# --------------------------
if __name__ == "__main__":
    
    with open(PERMUTATIONS_MAP_PATH, 'r') as json_file:
        permutations = json.load(json_file)
    
    ctx = multiprocessing.get_context('spawn')

    ############# RANDOM DATASET
    if EXPLORATION_MODE == 'dataset':
        if not MULTI_PROCS:
            # single process
            completed_runs = 0
            while completed_runs < NUM_VARIANTS[CIRCUIT_CLASS]:
                permutation = random.sample(list(permutations), 1)[0]
                del permutations[permutation]
                while os.path.isdir(f"{WD}/{OUTPUT_DIR[EXPLORATION_MODE]}/{permutation}"):
                    permutation = random.sample(list(permutations), 1)[0]
                baseline_created = create_place_route_baseline(permutation)
                
                if baseline_created:
                    variants_creation(permutation)
                    completed_runs += 1
        else:
            # multi processes
            manager = multiprocessing.Manager()
            permutation_queue = manager.Queue()
            counter_lock = manager.Lock()

            # Randomize and enqueue permutations
            shuffled_perms = random.sample(list(permutations), len(permutations))
            for perm in shuffled_perms:
                if not os.path.isdir(f"{WD}/{OUTPUT_DIR[EXPLORATION_MODE]}/{perm}"):
                    permutation_queue.put(perm)

            success_counter = manager.Value('i', 0)
            success_limit = NUM_VARIANTS[CIRCUIT_CLASS]

            with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PROCS, mp_context=ctx) as executor:
                # Start up to MAX_PROCS workers; each processes one permutation
                futures = [executor.submit(worker, permutation_queue, success_counter, success_limit, counter_lock)
                        for _ in range(len(shuffled_perms))]

                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            permutation, output = result
                            print(f"Permutation {permutation} processed with result: {output}")
                            completed += 1
                        if success_counter.value >= success_limit:
                            break  # Success limit met, no need to wait further
                    except Exception as exc:
                        print(f"A process generated an exception: {exc}")

            print(f"Total successful runs: {success_counter.value}")
    else:
        ############# RL
        if not MULTI_PROCS:
            # single process
            for i in range(MAX_RUNS[CIRCUIT_CLASS]):
                permutation = random.sample(list(permutations), 1)[0]
                train_loop_metagent(permutation, permutations, i)
        else:
            #multi processes
            with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PROCS,  mp_context=ctx) as executor:
            
                futures = {}
                for i in range(MAX_RUNS[CIRCUIT_CLASS]):
                    permutation = random.sample(list(permutations), 1)[0]

                    future = executor.submit(train_loop_metagent, permutation, permutations, i)
                    futures[future] = i

                for future in concurrent.futures.as_completed(futures):
                    i = futures[future]
                    try:
                        result = future.result()
                        print(f"Process {i} finished with result {result}")
                    except Exception as exc:
                        print(f"Process {i} generated an exception: {exc}")