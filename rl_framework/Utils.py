import re
import os
import shutil
from collections import deque
import torch
import numpy as np
import random


def set_seed(seed):
    """
    Set seed for all libraries to ensure reproducibility.
    
    Parameters:
    - seed (int): The seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seed set to: {seed}")


def _normalize_ele_stem(input_beam_file):
    """Return .ele file stem without extension."""
    return input_beam_file[:-4] if str(input_beam_file).endswith(".ele") else input_beam_file


def update_lattice_filename_in_ele(input_beam_file, elegant_input_filename):
    """
    Updates the lattice filename in the .ele tracking file.
    """
    input_beam_file = _normalize_ele_stem(input_beam_file)
    ele_file_path = input_beam_file + ".ele"
    
    if not os.path.exists(ele_file_path):
        print(f"Warning: {ele_file_path} does not exist. Skipping lattice filename update.")
        return
    
    try:
        with open(ele_file_path, 'r') as f:
            content = f.read()
        
        updated_content = re.sub(
            r'(\s*lattice\s*=\s*).*?(\s*(?:&|\n|$))',
            rf'\1{elegant_input_filename}\2',
            content,
            flags=re.IGNORECASE
        )
        
        with open(ele_file_path, 'w') as f:
            f.write(updated_content)
        
        print(f"Updated lattice filename in {ele_file_path} to: {elegant_input_filename}")
    
    except Exception as e:
        print(f"Error updating lattice filename in {ele_file_path}: {e}")


def update_results_path_in_ele(input_beam_file, results_path):
    """
    Updates all results folder references in a .ele tracking file.
    """
    input_beam_file = _normalize_ele_stem(input_beam_file)
    ele_file_path = input_beam_file + ".ele"
    if not os.path.exists(ele_file_path):
        print(f"Warning: {ele_file_path} does not exist. Skipping results path update.")
        return

    results_folder = str(results_path).rstrip("/\\")
    if not results_folder:
        return

    try:
        with open(ele_file_path, "r") as f:
            content = f.read()

        updated_content = re.sub(
            r'(?<![\w/])results(?:_[^/\s"]+)?/',
            f"{results_folder}/",
            content
        )

        with open(ele_file_path, "w") as f:
            f.write(updated_content)

    except Exception as e:
        print(f"Error updating results path in {ele_file_path}: {e}")


def update_random_seed_in_ele(input_beam_file, seed):
    """
    Updates random_number_seed in a .ele tracking file.
    """
    input_beam_file = _normalize_ele_stem(input_beam_file)
    ele_file_path = input_beam_file + ".ele"
    if not os.path.exists(ele_file_path):
        print(f"Warning: {ele_file_path} does not exist. Skipping random seed update.")
        return

    try:
        with open(ele_file_path, "r") as f:
            content = f.read()

        updated_content = re.sub(
            r"(\s*random_number_seed\s*=\s*)\d+",
            rf"\g<1>{int(seed)}",
            content,
            flags=re.IGNORECASE
        )

        with open(ele_file_path, "w") as f:
            f.write(updated_content)

    except Exception as e:
        print(f"Error updating random seed in {ele_file_path}: {e}")


def create_seeded_tracking_file(input_beam_file, elegant_input_filename, results_path, seed):
    """
    Creates a seed-specific copy of the tracking .ele file and updates:
    - lattice filename
    - results folder paths
    - random_number_seed

    Returns:
    - seeded_input_beam_file (str): file stem without extension, e.g. 'track_800'
    """
    input_beam_file = _normalize_ele_stem(input_beam_file)
    source_ele_path = f"{input_beam_file}.ele"
    seeded_input_beam_file = f"{input_beam_file}_{seed}"
    seeded_ele_path = f"{seeded_input_beam_file}.ele"

    if not os.path.exists(source_ele_path):
        raise FileNotFoundError(f"Tracking file not found: {source_ele_path}")

    shutil.copy2(source_ele_path, seeded_ele_path)
    update_lattice_filename_in_ele(seeded_input_beam_file, elegant_input_filename)
    update_results_path_in_ele(seeded_input_beam_file, results_path)
    update_random_seed_in_ele(seeded_input_beam_file, seed)

    return seeded_input_beam_file


#remove alll watch points that we didn't add
# and update the line definitions accordingly
# Save the original file content into 'unchanged_input.lte'
# This function removes all 'watch' elements from the .lte file and updates line definitions accordingly.
def remove_watch_points(file_path):
    """
    Removes all 'watch' elements from the .lte file and updates line definitions accordingly.
    Saves the original file content into 'unchanged_input.lte'.

    :param file_path: Path to the .lte file
    """
    # Save the original file content
    with open(file_path, 'r') as file:
        original_content = file.read()
    with open("unchanged_input.lte", 'w') as backup_file:
        backup_file.write(original_content)

    # Read the file and process its content
    with open(file_path, 'r') as file:
        lines = file.readlines()

    updated_lines = []
    watch_elements = set()  # To store all watch element names

    # First pass: Remove watch elements and collect their names
    for line in lines:
        if re.match(r"^\s*\w+\s*:\s*watch", line, re.IGNORECASE):
            match = re.match(r"^\s*(\w+)\s*:", line)
            if match:
                watch_elements.add(match.group(1))
        else:
            updated_lines.append(line)

    # Second pass: Update line definitions to remove watch elements
    final_lines = []
    for line in updated_lines:
        line_match = re.match(r"^\s*(\w+)\s*:\s*LINE\s*=\s*\((.*)\)", line, re.IGNORECASE)
        if line_match:
            line_name, elements = line_match.groups()
            elements_list = [e.strip() for e in elements.split(",")]
            filtered_elements = [e for e in elements_list if e not in watch_elements]
            final_lines.append(f"{line_name}: LINE=({', '.join(filtered_elements)})")
        else:
            final_lines.append(line.strip())

    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.write("\n".join(final_lines) + "\n")


def parse_lattice_file(file_path):
    """Parses a lattice file and extracts relevant elements into a dictionary."""
    elements = {}

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("!"):  # Skip empty lines and comments
                continue
            
            match = re.match(r"(\w+):\s*(\w+),\s*(.*)", line)
            if match:
                name, element_type, params = match.groups()

                # Identify element type
                if "WATCH" in element_type.upper():
                    category = "watch point"
                elif "QUAD" in element_type.upper():
                    category = "quadrupole magnet"
                elif "SBEND" in element_type.upper():
                    category = "dipole magnet"
                elif "DRIFT" in element_type.upper():
                    category = "drift"
                elif "MAXAMP" in element_type.upper():  
                    continue  # Skip apertures

                else:
                    continue  # Skip other elements

                # Extract parameters
                param_dict = {}
                for param in params.split(","):
                    key_value = param.split("=")
                    if len(key_value) == 2:
                        key, value = key_value
                        param_dict[key.strip()] = value.strip()
                
                elements[name] = {"type": category, "parameters": param_dict}

    return elements



'''
1. Watch points are only added after QUAD and SBEND elements.
2. The watch points are stored correctly and used sequentially for each repeated element instance.
3. LINE definitions are updated with the correct watch points in their correct positions.
'''

def add_watch_points(input_file, output_file, results_path):
    """
    Adds a watch point after each repeated quadrupole (QUAD) or dipole (SBEND) magnet in the lattice file,
    while preserving and updating LINE definitions to include only the generated watch points.
    """
    element_counts = {}  # Store element counts for QUAD and SBEND elements
    line_definitions = {}  # Store LINE definitions
    updated_lines = []  # Store intermediate lines without modification
    watch_points = []  # Store generated watch points
    controllable_elements = []  # Elements to add watch points after (QUAD, SBEND)
    watch_point_queues = {}  # Dictionary mapping elements to queues of watch points

    # First pass: Identify and count occurrences of QUAD and SBEND elements in LINE definitions
    with open(input_file, "r") as file:
        for line in file:
            line = line.rstrip()
            
            # Capture LINE definitions
            line_match = re.match(r"(\w+):\s*LINE\s*=\s*\((.*)\)", line)
            if line_match:
                line_name, elements = line_match.groups()
                elements_list = [e.strip() for e in elements.split(",")]
                line_definitions[line_name] = elements_list
                
                # Count occurrences only for QUAD and SBEND elements
                for element in elements_list:
                    if re.match(r"(\w+)", element):
                        element_counts[element] = element_counts.get(element, 0) + 1
                continue  # Store for later modification
            
            updated_lines.append(line)
    
    # Second pass: Modify the input file and add watch points
    final_lines = []
    
    for line in updated_lines:
        line = line.rstrip()
        
        # Match QUAD or SBEND elements
        match = re.match(r"(\w+):\s*(QUAD|SBEND)", line)
        if match:
            element_name = match.group(1)
            controllable_elements.append(element_name)
            
            # Add watch points based on element count
            if element_name in element_counts:
                count = element_counts[element_name]
                watch_point_queues[element_name] = deque()
                
                for i in range(1, count + 1):
                    watch_name = f"W{element_name}_{i}"
                    watch_line = f"{watch_name}: WATCH, filename=\"{results_path}{watch_name}.sdds\", mode=coord"
                    watch_points.append(watch_name)
                    final_lines.append(watch_line)
                    watch_point_queues[element_name].append(watch_name)
        
        final_lines.append(line)
    
    # Update LINE definitions with new watch points in correct positions
    for line_name, elements in line_definitions.items():
        new_elements_list = []
        
        for element in elements:
            if element in watch_point_queues and watch_point_queues[element]:
                new_elements_list.append(watch_point_queues[element].popleft())
            new_elements_list.append(element)
        
        final_lines.append(f"{line_name}: LINE=({', '.join(new_elements_list)})")
    
    # Write the updated lines to the output file
    with open(output_file, "w") as file:
        file.write("\n".join(final_lines) + "\n")

def add_final_watch_point(input_file, output_file, beamline_name,results_path):
    """
    Adds a new watch point named 'final_WP' to the specified beamline in the lattice file.
    The new watch point is defined at the beginning of the file.

    Parameters:
        input_file (str): Path to the input file containing the lattice definitions.
        output_file (str): Path to the output file where the modified content will be saved.
        beamline_name (str): The name of the beamline to which the watch point should be added.
    """
    # Define the new watch point
    new_watch_point = f'final_WP: WATCH, filename="{results_path}final_WP.sdds", mode=coord'

    # Read the input file
    with open(input_file, "r") as file:
        lines = file.readlines()

    # Initialize variables
    updated_lines = []
    beamline_updated = False
    watch_point_added = False

    # Process each line
    for i, line in enumerate(lines):
        line = line.rstrip()

        # Add the new watch point definition at the beginning of the file
        if not watch_point_added and not line.startswith("!"):
            updated_lines.append(new_watch_point)
            watch_point_added = True

        # Check if the line defines the specified beamline
        if line.startswith(f"{beamline_name}: LINE=(") and not beamline_updated:
            # Find the closing parenthesis and append 'final_WP' before it
            closing_paren_index = line.rfind(")")
            if closing_paren_index != -1:
                line = line[:closing_paren_index] + ", final_WP" + line[closing_paren_index:]
                beamline_updated = True

        updated_lines.append(line)

    # Write the updated content to the output file
    with open(output_file, "w") as file:
        file.write("\n".join(updated_lines) + "\n")

def change_initial_content(file_path, output_file):
    variables = []
    # Read the input file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Patterns to identify controllable elements and attributes
    quad_pattern = re.compile(r"(\w+)\s*:\s*QUAD\s*,\s*L\s*=\s*([\d\.eE+-]+)\s*,\s*K1\s*=\s*([\d\.eE+-]+)\s*,\s*HKICK\s*=\s*([\d\.eE+-]+)\s*,\s*VKICK\s*=\s*([\d\.eE+-]+)")#r"^(w+):\s*QUAD,.*?K1\s*=\s*([-\d.eE]+),\s+HKICK\s*=\s*([-\d.eE]+),\s+VKICK=([-\d.eE]+)")
    sbend_pattern = re.compile(r"(\w+)\s*:\s*SBEND\s*,\s*L\s*=\s*([\d\.eE+-]+)\s*,\s*ANGLE\s*=\s*([\d\.eE+-]+)(?:\s*,\s*FSE\s*=\s*([\d\.eE+-]+))?")

    # Store modified lines
    modified_lines = []

    for line in lines:
        # Check for QUAD elements and modify K1, HKICK, and VKICK
        quad_match = quad_pattern.search(line)
        if quad_match:
            element, L, K1, HKICK, VKICK  = quad_match.groups()
            new_line = (
                f"{element}:  QUAD,L={L}, "
                f"K1={{variables['{element}K1']}}, "
                f"HKICK={{variables['{element}HKICK']}}, "
                f"VKICK={{variables['{element}VKICK']}}\n"
            )
            modified_lines.append(new_line)
            variables.append(f"{element}K1")
            variables.append(f"{element}HKICK")
            variables.append(f"{element}VKICK")
            continue

        # Check for SBEND elements and modify FSE
        sbend_match = sbend_pattern.search(line)
        if sbend_match:
            element, L, ANGLE, FSE = sbend_match.groups()
            new_line = (
                f"{element}:    SBEND,  L={L},      ANGLE={ANGLE}, "
                f"FSE={{variables['{element}FSE']}}\n"
            )
            modified_lines.append(new_line)
            variables.append(f"{element}FSE")
            continue

        # If no match, keep the line unchanged
        modified_lines.append(line)

    # Join all lines to form the final content
    modified_content = "".join(modified_lines)

    with open(output_file, "w") as file:
        file.write("\n".join(modified_lines) + "\n")
    return modified_content, variables

def create_dict_from_lists(keys, values):
    """
    Creates a dictionary from two lists: one for keys and one for values.
    If the lists have different lengths, the extra elements are ignored.
    
    :param keys: List of keys for the dictionary
    :param values: List of values for the dictionary
    :return: Dictionary with keys from the first list and values from the second list
    """
    # Use zip to pair keys and values, then convert to a dictionary
    return dict(zip(keys, values))


####
#functions for creating a graph from the lattice file
####

import re
from collections import deque, defaultdict

def parse_lte_file(file_path):
    """
    Parses the .lte file and returns a dictionary of element definitions.
    """
    element_definitions = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('!'):
                match = re.match(r"(\w+):\s*(.*)", line)
                if match:
                    element_name, element_definition = match.groups()
                    element_definitions[element_name] = element_definition
    return element_definitions

def expand_beamline(element_definitions, beamline_name):
    """
    Recursively expands the beamline definition to include only base elements.
    """
    def expand(element):
        if element in element_definitions:
            definition = element_definitions[element]
            if definition.startswith('LINE'):
                elements = re.findall(r"\w+", definition)
                expanded_elements = []
                for elem in elements:
                    expanded_elements.extend(expand(elem))
                return expanded_elements
            else:
                return [(element, definition)]
        else:
            return [(element, '')]

    expanded_beamline = expand(beamline_name)
    return expanded_beamline

def create_graph(expanded_beamline):
    """
    Creates a graph-like structure from the expanded beamline.
    """
    """graph = []
    for element, definition in expanded_beamline:
        element_type = definition.split(',')[0] if definition else 'UNKNOWN'
        attributes = {}
        if definition:
            attributes = dict(re.findall(r"(\w+)=([\w\.\+\-]+)", definition))
        graph.append({
            'name': element,
            'type': element_type,
            'attributes': attributes
        })"""
    graph = []
    for element in expanded_beamline:
        element_name = element[0] if element else 'UNKNOWN'
        #print("element_name", element_name)
        attributes = {}
        if element[1]:
            # Replace tab characters before extracting attributes
            element_list = element[1].split(',')
            element_type= element_list[0] if element_list[0] else 'UNKNOWN'
            element_attributes =element_list[1:]
            clean_str = str(element_attributes).replace('\\t', '')
            
            attributes = dict(re.findall(r"(\w+)\s*=\s*([\w\.\+\-eE]+)", clean_str))
        else:
            element_type= "UNKNOWN"
            attributes= None
        graph.append({
            'name': element_name,
            'type': element_type,
            'attributes': attributes
        })
    return graph

def process_lte_file_to_graph(file_path, beamline_name):
    """
    Processes the .lte file and returns the expanded beamline graph.
    """
    element_definitions = parse_lte_file(file_path)
    expanded_beamline = expand_beamline(element_definitions, beamline_name)
    graph = create_graph(expanded_beamline)
    return graph

#####
#functions for creating a feature matrix from the graph
# the main issue is the size of the feature matrix depends on the number of unique element
#####
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

'''
Explanation:
1.	Define a Fixed Set of Features: We define a fixed set of possible element types and initialize a OneHotEncoder to encode these types as numerical vectors.
2.	Collect Attribute Names: We define the attribute names that we care about, such as 'L', 'K1', 'HKICK', 'VKICK', 'ANGLE', 'FSE', 'X_MAX', 'Y_MAX', and 'ELLIPTICAL'.
3.	Encode Element Types: For each element in the graph, we use the OneHotEncoder to encode the element type as a numerical vector.
4.	Normalize Numerical Attributes: We collect the attribute values for each element and normalize them using a MinMaxScaler.
5.	Create the Feature Matrix: We combine the encoded element types and normalized attributes into a single feature matrix.
6.	Example Usage: We demonstrate how to use the create_feature_matrix function with a sample graph.
This implementation should provide a feature matrix that can be used as input to a neural network. Each row in the matrix represents an element, and each column represents a feature. The feature matrix can be further processed or fed directly into a neural network for training or inference.
'''

def create_feature_matrix(graph):
    # Define the set of possible element types
    element_types = ['charge', 'WATCH', 'MAXAMP', 'rfca','QUAD', 'SBEND', 'DRIFT', 'UNKNOWN']
    
    # Initialize the OneHotEncoder for element types
    type_encoder = OneHotEncoder(categories=[element_types], handle_unknown='ignore')
    type_encoder.fit(np.array(element_types).reshape(-1, 1))
    
    # Initialize the MinMaxScaler for numerical attributes
    scaler = MinMaxScaler()
    
    # Collect all attribute names
    attribute_names = ['L', 'K1', 'HKICK', 'VKICK', 'ANGLE', 'FSE', 'X_MAX', 'Y_MAX', 'ELLIPTICAL']
    
    # Create the feature matrix
    feature_matrix = []
    for node in graph:
        # Encode the element type
        element_type = node['type']
        type_vector = type_encoder.transform([[element_type]]).toarray().flatten()
        
        # Collect and normalize the attributes
        attributes = []
        for attr in attribute_names:
            value = node['attributes'].get(attr, 0)
            attributes.append(float(value))
        
        # Combine the type vector and attributes
        feature_vector = np.concatenate([type_vector, attributes])
        feature_matrix.append(feature_vector)
    
    # Convert the feature matrix to a numpy array
    feature_matrix = np.array(feature_matrix)
    
    # Normalize the numerical attributes
    if feature_matrix.shape[1] > len(element_types):
        feature_matrix[:, len(element_types):] = scaler.fit_transform(feature_matrix[:, len(element_types):])
    
    return feature_matrix


# To preprocess the observation data and compute the covariance matrix of the standardized data.
# The function should return the covariance matrix of the standardized data and the column means of the original observation data.
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def compute_covariance_matrix_mean(observation):
    """
    Computes the covariance matrix of the standardized observation data,
    handling cases where the number of observations is too small.

    Parameters:
    observation (pd.DataFrame): The observation data as a DataFrame.

    Returns:
    tuple: (covariance matrix, column means)
    """
    x, features = observation.shape

    if x < 3:  # Ensure we have enough samples
        print(f"Warning: Too few samples ({x}). Returning identity matrix as fallback.")
        return np.identity(features), observation.mean()

    # Standardize the data
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(observation), columns=observation.columns)

    # Compute covariance matrix
    cov_matrix_standardized = df_standardized.cov()

    # Add a small regularization term to avoid singular matrix issues
    cov_matrix_standardized += np.eye(features) * 1e-6

    return cov_matrix_standardized, observation.mean()


###
#functions for logging the environment
####
import csv

def setLogger(load, log_file= 'environment_log.csv', headers= ('reward', 'actions', 'duration')):
    """
    Set up the logger for the environment.
    """
    if load== True:
        file_handler = open(log_file, "at")
        logger = csv.DictWriter(file_handler, fieldnames=headers)
    else:
        file_handler = open(log_file, "wt")
        logger = csv.DictWriter(file_handler,
                                  fieldnames= headers )
        logger.writeheader()
    file_handler.flush()
    return logger, file_handler

def reset_specific_keys(input_dict):
    """
    Sets the values of keys containing 'HKICK', 'VKICK', or 'FSE' to zero.

    Args:
        input_dict (dict): The input dictionary.

    Returns:
        dict: The updated dictionary with specific keys reset to zero.
    """
    for key in input_dict.keys():
        if any(substring in key for substring in ["HKICK", "VKICK", "FSE"]):
        #if any(substring in key for substring in ["HKICK", "VKICK"]):
            input_dict[key] = 0
    return input_dict



# for generalizing over the number of initial particles, this function changes the initial number of particles everytime we reset the environment
def change_num_initial_particles(path="", num=10000):
    with open(path, "r") as f:
        lines = f.readlines()
    k = None
    pattern = re.compile(r"^\s*n_particles_per_bunch\s*=\s*(\d+)")
    for i, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            k = int(match.group(1))
            lines[i] = re.sub(r"(\d+)", str(num), line, count=1)
            break
    with open(path, "w") as f:
        f.writelines(lines)
    return k, num

#expected sources ["Random", "Memory", "Values"]
def run_episode(env, source= "Random", actions= None, memory_episode= None):
    state = env.reset()
    done = False
    if source== "Random":
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _, info = env.step(action)
            state= next_state
            print(info['output_file'] ,reward, done, env.get_number_of_particles())
            if done:
                break
    elif source== "Memory":
        if memory_episode is not None:
            for i in range(len(memory_episode)):
                next_state, reward, done, _, info = env.step(list(memory_episode[i].action), convert=True)
                state= next_state
                print(info['output_file'] ,reward, done, env.get_number_of_particles())
                if done:
                    break
        else:
            print("No memory episode found.")
    elif source== "Values":
        if actions is not None:
            i=0  
            while not done:
                count= env._check_number_of_variables_to_be_set_at_this_iteration()
                if count== 3:
                    action_temp= np.array(actions[i :i+count])
                    filler = np.array([0.0])
                    action = np.concatenate((action_temp, filler), axis=0)
                else:
                    action_temp= np.array(actions[i :i+count])
                    filler = np.array([0.0, 0.0, 0.0])
                    action = np.concatenate((filler, action_temp), axis=0)
                i= i+ count
                next_state, reward, done, _, info = env.step(action)
                print(info['output_file'] ,info['number_of_particles'],reward, done, env.get_number_of_particles())
                state= next_state
                """if done:
                    break"""
        else:
            print("No actions found.")
    else:
        print("Invalid source.")


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def find_maxamp_for_watch_points(element, graph, watch_points):
    """
    Find the MAXAMP element immediately preceding a given watch point in the graph.

    Parameters:
    - element (str): Name of the watch point (e.g., 'WQ1L0_1').
    - graph (list): List of dictionaries, each containing 'name', 'type', and 'attributes'.
    - watch_points (list): List of watch point names in chronological order.

    Returns:
    - dict or None: The MAXAMP element (dictionary) immediately before the watch point,
      or None if no MAXAMP is found or the element is invalid.
    """
    if element not in watch_points:
        return None

    last_maxamp = None
    for item in graph:
        if item['type'] == 'MAXAMP':
            last_maxamp = item
        if item['name'] == element and item['type'] == 'WATCH':
            return last_maxamp
    return None

'''def find_next_maxamp_for_watch_points(element, graph, watch_points):
    """
    Find the MAXAMP element immediately preceding a given watch point in the graph.

    Parameters:
    - element (str): Name of the watch point (e.g., 'WQ1L0_1').
    - graph (list): List of dictionaries, each containing 'name', 'type', and 'attributes'.
    - watch_points (list): List of watch point names in chronological order.

    Returns:
    - dict or None: The MAXAMP element (dictionary) immediately before the watch point,
      or None if no MAXAMP is found or the element is invalid.
    """
    if element not in watch_points:
        return None

    Found= False
    watch= False
    i=0
    next_maxamp = None
    while not Found:
        item= graph[i]
        if item['name'] == element and item['type'] == 'WATCH':
            watch= True

        if item['type'] == 'MAXAMP' and watch:
            next_maxamp = item
            Found= True
        i=i+1

        return next_maxamp
    return None'''

def find_next_maxamp_for_watch_points(element, graph, watch_points):
    """
    Find the MAXAMP element immediately preceding a given watch point in the graph.

    Parameters:
    - element (str): Name of the watch point (e.g., 'WQ1L0_1').
    - graph (list): List of dictionaries, each containing 'name', 'type', and 'attributes'.
    - watch_points (list): List of watch point names in chronological order.

    Returns:
    - dict or None: The MAXAMP element (dictionary) immediately before the watch point,
      or None if no MAXAMP is found or the element is invalid.
    """
    if element not in watch_points:
        return None, None

     # Track the last MAXAMP element encountered
    last_maxamp = None
    watch= False
    next_maxamp = None

    
    for item in graph:

        # Check if the current item is a MAXAMP
        if item['type'] == 'MAXAMP' and not watch:
            last_maxamp = item
        # Check if the current item is the target watch point
        if item['name'] == element and item['type'] == 'WATCH':
            watch= True
            continue

        if watch and item['type'] == 'WATCH':
            next_maxamp = last_maxamp
            return last_maxamp, next_maxamp

        if (item['type'] == 'MAXAMP' ) and watch:
            next_maxamp = item
            return last_maxamp, next_maxamp
        
    if watch and item['name'] == 'final_WP':
        return last_maxamp, last_maxamp
     # Return None if the watch point is not found or no MAXAMP precedes it
    return None, None

def create_nn_representation(df, a, b, n_bins=5):
    """
    Create a fixed-size feature vector for neural network input from particle data.
    Combines robust statistical summaries, 2D histogram of (x, y), point count,
    fraction of points inside an elliptical region, covariance matrix for x, y, xp, yp,
    and ellipse parameters. Handles empty or small datasets and is robust to outliers.

    Parameters:
    - df: pandas DataFrame with columns ['x', 'y', 'xp', 'yp']
    - a: semi-major axis of the ellipse (x-axis)
    - b: semi-minor axis of the ellipse (y-axis)
    - n_bins: number of bins per dimension for 2D histogram (default: 5)

    Returns:
    - 1D numpy array with:
        - Median, IQR, 10th, 90th percentiles for x, y, xp, yp (4 * 4 = 16)
        - 2D histogram bin counts for (x, y) (n_bins * n_bins = 25 for n_bins=5)
        - Number of points (1)
        - Fraction of points inside ellipse (1)
        - Covariance matrix upper triangle for x, y, xp, yp (10 elements)
        - Ellipse parameters a, b (2)
        Total size: 16 + 25 + 1 + 1 + 10 + 2 = 55
    """
    output_size = 16 + n_bins**2 + 1 + 1 + 10 + 2
    if df.empty:
        return np.zeros(output_size)

    features = ['x', 'y', 'xp', 'yp']
    data = df[features].to_numpy()

    stats = []
    for i in range(data.shape[1]):
        col = data[:, i]
        if len(col) > 0:
            median = np.median(col)
            p10, p90 = np.percentile(col, [10, 90])
            iqr = np.percentile(col, 75) - np.percentile(col, 25) if len(col) > 1 else 0.0
        else:
            median, iqr, p10, p90 = 0.0, 0.0, 0.0, 0.0
        stats.extend([median, iqr, p10, p90])

    if len(df) > 0:
        x_range = np.percentile(df['x'], [1, 99]) if df['x'].std() > 0 else [-1, 1]
        y_range = np.percentile(df['y'], [1, 99]) if df['y'].std() > 0 else [-1, 1]
        hist, xedges, yedges = np.histogram2d(
            df['x'], df['y'], bins=n_bins, range=[x_range, y_range]
        )
        hist = hist.flatten() / len(df) if len(df) > 0 else np.zeros(n_bins**2)
    else:
        hist = np.zeros(n_bins**2)

    point_count = len(df)
    fraction_inside = points_in_region(df, a, b) / 100 if not df.empty else 0.0

    if len(df) < 3:
        cov_matrix = np.identity(4)
    else:
        scaler = StandardScaler()
        data_standardized = scaler.fit_transform(data)
        cov_matrix = np.cov(data_standardized, rowvar=False)
        cov_matrix += np.eye(4) * 1e-6
    cov_indices = np.triu_indices(4)
    cov_elements = cov_matrix[cov_indices]

    ellipse_params = np.array([a, b])

    return np.concatenate([stats, hist, [point_count, fraction_inside], cov_elements, ellipse_params])

def create_nn_representation_new(df, a_prev, b_prev, a_next, b_next, n_bins=5, initialNumParticles= 0):
    """
    Create a fixed-size feature vector for neural network input from particle data.
    Combines robust statistical summaries, 2D histogram of (x, y), point count,
    fraction of points inside an elliptical region, covariance matrix for x, y, xp, yp,
    and ellipse parameters. Handles empty or small datasets and is robust to outliers.

    Parameters:
    - df: pandas DataFrame with columns ['x', 'y', 'xp', 'yp']
    - a: semi-major axis of the ellipse (x-axis)
    - b: semi-minor axis of the ellipse (y-axis)
    - n_bins: number of bins per dimension for 2D histogram (default: 5)

    Returns:
    - 1D numpy array with:
        - Median, IQR, 10th, 90th percentiles for x, y, xp, yp (4 * 4 = 16)
        - 2D histogram bin counts for (x, y) (n_bins * n_bins = 25 for n_bins=5)
        - Number of points (1)
        - Fraction of points inside ellipse (1)
        - Covariance matrix upper triangle for x, y, xp, yp (10 elements)
        - Ellipse parameters a_previous, b_previous and a_next, b_next (4)
        Total size: 16 + 25 + 1 + 1 + 10 + 2 = 57
    """
    output_size = 16 + n_bins**2 + 1 + 1 + 10 + 4
    if df.empty:
        return np.zeros(output_size)

    features = ['x', 'y', 'xp', 'yp']
    data = df[features].to_numpy()

    stats = []
    for i in range(data.shape[1]):
        col = data[:, i]
        if len(col) > 0:
            median = np.median(col)
            p10, p90 = np.percentile(col, [10, 90])
            iqr = np.percentile(col, 75) - np.percentile(col, 25) if len(col) > 1 else 0.0
        else:
            median, iqr, p10, p90 = 0.0, 0.0, 0.0, 0.0
        stats.extend([median, iqr, p10, p90])

    if len(df) > 0:
        x_range = np.percentile(df['x'], [1, 99]) if df['x'].std() > 0 else [-1, 1]
        y_range = np.percentile(df['y'], [1, 99]) if df['y'].std() > 0 else [-1, 1]
        hist, xedges, yedges = np.histogram2d(
            df['x'], df['y'], bins=n_bins, range=[x_range, y_range]
        )
        hist = hist.flatten() / len(df) if len(df) > 0 else np.zeros(n_bins**2)
    else:
        hist = np.zeros(n_bins**2)

    point_count =len(df)/initialNumParticles
    fraction_inside = points_in_region(df, a_prev, b_prev) / 100 if not df.empty else 0.0

    if len(df) < 3:
        cov_matrix = np.identity(4)
    else:
        scaler = StandardScaler()
        data_standardized = scaler.fit_transform(data)
        cov_matrix = np.cov(data_standardized, rowvar=False)
        cov_matrix += np.eye(4) * 1e-6
    cov_indices = np.triu_indices(4)
    cov_elements = cov_matrix[cov_indices]

    ellipse_params = np.array([ a_prev, b_prev, a_next, b_next])

    return np.concatenate([stats, hist, [point_count, fraction_inside], cov_elements, ellipse_params])


def points_in_region(df, a, b):
    """
    Calculate the percentage of points within an elliptical region defined by
    semi-major axis a (x-axis) and semi-minor axis b (y-axis).
    Returns the percentage (0 to 100) or 0 if DataFrame is empty.
    """
    if df.empty:
        return 0.0

    in_region = (df['x']**2 / a**2 + df['y']**2 / b**2) <= 1
    percentage = (in_region.sum() / len(df)) * 100
    return percentage

def process_particle_data(file_path, watch_point, graph, watch_points, n_bins=5, initialNumParticles=0):
    """
    Process a particle data CSV file and return the neural network representation
    and the percentage of points within the specified elliptical region defined by
    the MAXAMP element preceding the watch point.

    Parameters:
    - file_path (str): Path to CSV file.
    - watch_point (str): Name of the watch point (e.g., 'WQ1L0_1').
    - graph (list): List of dictionaries with graph elements.
    - watch_points (list): List of watch point names in chronological order.
    - n_bins (int): Number of bins per dimension for 2D histogram (default: 5).

    Returns:
    - dict or None: Dictionary with neural network representation and percentage in region,
      or None if processing fails or no MAXAMP is found.
    """
    try:
        # Find the corresponding MAXAMP element
        #maxamp = find_maxamp_for_watch_points(watch_point, graph, watch_points)
        prev_maxamp ,next_maxamp = find_next_maxamp_for_watch_points(watch_point, graph, watch_points)
        
        if prev_maxamp is None or 'attributes' not in prev_maxamp or 'X_MAX' not in prev_maxamp['attributes'] or 'Y_MAX' not in prev_maxamp['attributes']:
            print(f"Error: No valid MAXAMP found for watch point {watch_point}")
            return None
        if next_maxamp is None or 'attributes' not in next_maxamp or 'X_MAX' not in next_maxamp['attributes'] or 'Y_MAX' not in next_maxamp['attributes']:
            print(f"Error: No valid MAXAMP found for watch point {watch_point}")
            return None

        # Extract X_MAX and Y_MAX as a and b
        a_prev = float(prev_maxamp['attributes']['X_MAX'])
        b_prev = float(prev_maxamp['attributes']['Y_MAX'])

         # Extract X_MAX and Y_MAX as a and b
        a_next = float(next_maxamp['attributes']['X_MAX'])
        b_next = float(next_maxamp['attributes']['Y_MAX'])

        # Read CSV file
        df = pd.read_csv(file_path)

        # Validate required columns
        required_columns = ['x', 'y', 'xp', 'yp']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("CSV file must contain columns: x, y, xp, yp")

        # Create neural network representation
        nn_representation = create_nn_representation_new(df, a_prev, b_prev, a_next, b_next, n_bins, initialNumParticles=initialNumParticles) #edit here 

        # Calculate percentage of points in elliptical region
        percentage_in_region = points_in_region(df, a_prev, b_prev)

        return {
            'nn_representation': nn_representation,
            'percentage_in_region': percentage_in_region,
            'a_prev': a_prev,
            'b_prev': b_prev,
            'a_next': a_next,
            'b_next': b_next
        }

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None