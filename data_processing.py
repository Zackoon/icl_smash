import os
import melee
import numpy as np
import pandas as pd
import zipfile
from tqdm import tqdm
import tempfile

def load_slp_file(file_path):
    """Initialize and connect to a SLP file."""
    console = melee.Console(
        is_dolphin=False,
        allow_old_version=True,
        path=file_path
    )
    console.connect()
    return console

def flatten_gamestate(frame):
    """Convert a single frame of gamestate into a flat dictionary."""
    flat = {
        "frame": frame["frame"],
        "distance": frame["distance"],
        "stage": frame["stage"]
    }

    # List of all button names we want to track
    button_names = [
        "BUTTON_A", "BUTTON_B", "BUTTON_X", "BUTTON_Y", "BUTTON_Z",
        "BUTTON_L", "BUTTON_R", "BUTTON_START",
        "BUTTON_D_UP", "BUTTON_D_DOWN", "BUTTON_D_LEFT", "BUTTON_D_RIGHT"
    ]

    # Process player data
    for port in sorted(frame["players"].keys()):
        p = frame["players"][port]
        prefix = f"p{port}_"

        # Flatten all the regular fields
        flat.update({
            prefix + "action": p["action"],
            prefix + "action_frame": p["action_frame"],
            prefix + "character": p["character"],
            prefix + "controller_c_stick_x": p["controller_state"]["c_stick_x"],
            prefix + "controller_c_stick_y": p["controller_state"]["c_stick_y"],
            prefix + "controller_l_shoulder": p["controller_state"]["l_shoulder"],
            prefix + "controller_r_shoulder": p["controller_state"]["r_shoulder"],
            prefix + "controller_main_stick_x": p["controller_state"]["main_stick_x"],
            prefix + "controller_main_stick_y": p["controller_state"]["main_stick_y"],
        })

        # Flatten the button states into individual boolean fields
        button_dict = p["controller_state"]["button"]
        for button_name in button_names:
            button_key = f"{prefix}button_{button_name.lower()}"
            # Convert the boolean value to int (0 or 1)
            button_value = int(button_dict.get(getattr(melee.Button, button_name), False))
            flat[button_key] = button_value

        # Add the rest of the player fields...
        flat.update({
            prefix + "ecb_bottom_x": p["ecb"]["bottom"][0],
            prefix + "ecb_bottom_y": p["ecb"]["bottom"][1],
            prefix + "ecb_left_x": p["ecb"]["left"][0],
            prefix + "ecb_left_y": p["ecb"]["left"][1],
            prefix + "ecb_right_x": p["ecb"]["right"][0],
            prefix + "ecb_right_y": p["ecb"]["right"][1],
            prefix + "ecb_top_x": p["ecb"]["top"][0],
            prefix + "ecb_top_y": p["ecb"]["top"][1],
            prefix + "facing": p["facing"],
            prefix + "hitlag_left": p["hitlag_left"],
            prefix + "hitstun_frames_left": p["hitstun_frames_left"],
            prefix + "iasa": p["iasa"],
            prefix + "invulnerability_left": p["invulnerability_left"],
            prefix + "invulnerable": p["invulnerable"],
            prefix + "jumps_left": p["jumps_left"],
            prefix + "moonwalkwarning": p["moonwalkwarning"],
            prefix + "nana": p["nana"],
            prefix + "off_stage": p["off_stage"],
            prefix + "on_ground": p["on_ground"],
            prefix + "percent": p["percent"],
            prefix + "position_x": p["position"]["x"],
            prefix + "position_y": p["position"]["y"],
            prefix + "shield_strength": p["shield_strength"],
            prefix + "speed_air_x_self": p["speed_air_x_self"],
            prefix + "speed_ground_x_self": p["speed_ground_x_self"],
            prefix + "speed_x_attack": p["speed_x_attack"],
            prefix + "speed_y_self": p["speed_y_self"],
            prefix + "stock": p["stock"]
        })

    # Process projectile data
    for i in range(2):  # Support up to 2 projectiles
        prefix = f"proj{i}_"
        if i < len(frame["projectiles"]):
            proj = frame["projectiles"][i]
            flat.update({
                prefix + "frame": proj["frame"],
                prefix + "owner": proj["owner"],
                prefix + "position_x": proj["position"]["x"],
                prefix + "position_y": proj["position"]["y"],
                prefix + "speed_x": proj["speed"]["x"],
                prefix + "speed_y": proj["speed"]["y"],
                # prefix + "subtype": proj["subtype"],  # type of projectile likely not necessary to know
                # prefix + "type": proj["type"]
            })
        else:
            for key in ["frame", "owner", "position_x", "position_y", "speed_x", "speed_y"]: # , "subtype", "type"]:
                flat[prefix + key] = None

    return flat

def process_gamestate(gamestate):
    """Convert a gamestate object into a structured dictionary."""
    if gamestate.menu_state != melee.enums.Menu.IN_GAME:
        return None

    frame_data = {
        "frame": gamestate.frame,
        "distance": gamestate.distance,
        "stage": gamestate.stage.name if gamestate.stage else None,
        "players": {},
        "projectiles": []
    }

    for port, player in gamestate.players.items():
        if player is None:
            continue
        frame_data["players"][port] = {
            "action": player.action.name if player.action else None,
            "action_frame": player.action_frame,
            "character": player.character.name if player.character else None,
            "controller_state": {
                "button": player.controller_state.button,
                "c_stick_x": player.controller_state.c_stick[0],
                "c_stick_y": player.controller_state.c_stick[1],
                "l_shoulder": player.controller_state.l_shoulder,
                "r_shoulder": player.controller_state.r_shoulder,
                "main_stick_x": player.controller_state.main_stick[0],
                "main_stick_y": player.controller_state.main_stick[1]
            },
            "ecb": {
                "bottom": tuple(player.ecb_bottom) if player.ecb_bottom else (None, None),
                "left": tuple(player.ecb_left) if player.ecb_left else (None, None),
                "right": tuple(player.ecb_right) if player.ecb_right else (None, None),
                "top": tuple(player.ecb_top) if player.ecb_top else (None, None),
            },
            "facing": player.facing,
            "hitlag_left": player.hitlag_left,
            "hitstun_frames_left": player.hitstun_frames_left,
            "iasa": player.iasa,
            "invulnerability_left": player.invulnerability_left,
            "invulnerable": player.invulnerable,
            "jumps_left": player.jumps_left,
            "moonwalkwarning": player.moonwalkwarning,
            "nana": player.nana is not None,
            "off_stage": player.off_stage,
            "on_ground": player.on_ground,
            "percent": player.percent,
            "position": {
                "x": player.position.x if player.position else None,
                "y": player.position.y if player.position else None
            },
            "shield_strength": player.shield_strength,
            "speed_air_x_self": player.speed_air_x_self,
            "speed_ground_x_self": player.speed_ground_x_self,
            "speed_x_attack": player.speed_x_attack,
            "speed_y_self": player.speed_y_self,
            "stock": player.stock
        }

    return frame_data

def process_slp_file(file_path, sample_freq=15):
    """Process a SLP file and return sampled numpy array."""
    console = load_slp_file(file_path)
    flattened_frames = []

    while True:
        gamestate = console.step()
        if gamestate is None:
            break

        frame_data = process_gamestate(gamestate)
        if frame_data is not None:
            flattened_frames.append(flatten_gamestate(frame_data))

    # Convert to DataFrame and numpy array
    df = pd.DataFrame(flattened_frames)
    data = df.to_numpy()
    columns = df.columns.tolist()

    # Get all port-based enum columns that exist in the data
    enum_cols = ['stage']  # Start with stage
    for port in range(1, 5):  # Ports 1-4
        action_col = f'p{port}_action'
        char_col = f'p{port}_character'
        if action_col in columns:
            enum_cols.append(action_col)
        if char_col in columns:
            enum_cols.append(char_col)

    # Get indices of enum columns and other columns
    enum_indices = [columns.index(col) for col in enum_cols]
    other_indices = [i for i in range(len(columns)) if i not in enum_indices]

    # Reorder the columns, such that the enum columns are moved to the end
    reordered_data = np.concatenate([data[:, other_indices], data[:, enum_indices]], axis=1)
    reordered_columns = [columns[i] for i in other_indices] + [columns[i] for i in enum_indices]
    
    # Sample the data
    sampled_idxs = np.arange(0, reordered_data.shape[0], sample_freq)
    sampled_data = reordered_data[sampled_idxs]
    
    return sampled_data, reordered_columns

def save_processed_data(sampled_data, columns, output_dir, base_filename, save_csv=False):
    """Save processed data and column names.
    
    Args:
        sampled_data: numpy array of processed game data
        columns: list of column names
        output_dir: directory to save the files
        base_filename: base name for the output files
        save_csv: if True, also save as CSV (default: False)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save numpy arrays
    np.save(os.path.join(output_dir, f"{base_filename}_data.npy"), sampled_data)
    np.save(os.path.join(output_dir, f"{base_filename}_columns.npy"), np.array(columns))
    
    # Save as CSV if requested
    if save_csv:
        df = pd.DataFrame(sampled_data, columns=columns)
        csv_path = os.path.join(output_dir, f"{base_filename}_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV to: {csv_path}")

def create_sequences(data, input_len=10, target_len=5):
    """Create input and target sequences from data.
    
    Args:
        data: numpy array of processed game data
        input_len: length of input sequences
        target_len: length of target sequences
    
    Returns:
        tuple: (input_sequences, target_sequences)
    """
    input_seqs = []
    target_seqs = []
    for t in range(len(data) - input_len - target_len):
        input_seqs.append(data[t : t + input_len])
        target_seqs.append(data[t + input_len : t + input_len + target_len])
    
    return np.array(input_seqs), np.array(target_seqs)

def save_sequences(input_seqs, target_seqs, output_dir, base_filename):
    """Save input and target sequences to compressed npz file.
    
    Args:
        input_seqs: numpy array of input sequences
        target_seqs: numpy array of target sequences
        output_dir: directory to save the file
        base_filename: base name for the output file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base_filename}_sequences.npz")
        
    # Use compressed format
    np.savez_compressed(output_path, inputs=input_seqs, targets=target_seqs)
    
    print(f"Saved sequences to {output_path}")
    print(f"Input shape: {input_seqs.shape}")
    print(f"Target shape: {target_seqs.shape}")

def process_and_save_sequences(data, output_dir, base_filename, input_len=10, target_len=5):
    """Process data into sequences and save them.
    
    Args:
        data: numpy array of processed game data
        output_dir: directory to save the sequences
        base_filename: base name for the output file
        input_len: length of input sequences
        target_len: length of target sequences
    
    Returns:
        tuple: (input_sequences, target_sequences)
    """
    input_seqs, target_seqs = create_sequences(data, input_len, target_len)
    save_sequences(input_seqs, target_seqs, output_dir, base_filename)
    return input_seqs, target_seqs

def main(slp_file_path, output_dir, base_filename, sample_freq=15, save_csv=False):
    """Main function to process a SLP file and save the results.
    
    Args:
        slp_file_path: path to the .slp file
        output_dir: directory to save the processed data
        base_filename: base name for the output files
        sample_freq: frequency of frame sampling (default: 15)
        save_csv: if True, also save as CSV (default: False)
    """
    sampled_data, columns = process_slp_file(slp_file_path, sample_freq)
    save_processed_data(sampled_data, columns, output_dir, base_filename, save_csv=save_csv)
    return sampled_data.shape

def process_zip_files_streaming(zip_path, output_dir, input_len=10, target_len=5):
    """Process multiple .slp files from a zip file one at a time.
    
    Args:
        zip_path: path to the zip file containing .slp files
        output_dir: directory to save the processed sequences
        input_len: length of input sequences
        target_len: length of target sequences
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get all .slp files from zip
        slp_files = [f for f in zip_ref.namelist() if f.endswith('.slp')]
        
        # Process one file at a time
        for zip_path in tqdm(slp_files):
            try:
                base_filename = os.path.splitext(os.path.basename(zip_path))[0]
                
                # Create a temporary file that is automatically cleaned up
                with tempfile.NamedTemporaryFile(suffix='.slp', delete=True) as temp_file:
                    with zip_ref.open(zip_path) as source:
                        temp_file.write(source.read())
                    temp_file.flush()
                    
                    # Process the file
                    sampled_data, columns = process_slp_file(temp_file.name)
                    
                    # Create and save sequences
                    process_and_save_sequences(
                        sampled_data,
                        output_dir,
                        base_filename,
                        input_len=input_len,
                        target_len=target_len
                    )
                    
            except Exception as e:
                print(f"Error processing {zip_path}: {str(e)}")
                continue

    print(f"Processed {len(slp_files)} files in total")

# Example usage:
if __name__ == "__main__":
    slp_file = "./data/Stream-Game_20220828T223051.slp"
    output_dir = "./processed_data"
    base_filename = "example1"
    
    # Process and save in both .npy and .csv formats
    shape = main(slp_file, output_dir, base_filename, save_csv=True)
    print(f"Processed data shape: {shape}")
