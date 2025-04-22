import os
import melee
import numpy as np
import pandas as pd
import zipfile
import py7zr
import tempfile
import os
from pathlib import Path
from typing import Union, List
from tqdm import tqdm
from glob import glob
from typing import Tuple

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

    # List of all button names we want to track (removed D-pad buttons)
    button_names = [
        "BUTTON_A", "BUTTON_B", "BUTTON_X", "BUTTON_Y", "BUTTON_Z",
        "BUTTON_L", "BUTTON_R", "BUTTON_START"
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
            prefix + "facing": p["facing"],
            prefix + "hitlag_left": p["hitlag_left"],
            prefix + "hitstun_frames_left": p["hitstun_frames_left"],
            prefix + "jumps_left": p["jumps_left"],
            prefix + "off_stage": p["off_stage"],
            prefix + "on_ground": p["on_ground"],
            prefix + "percent": p["percent"],
            prefix + "position_x": p["position"]["x"],
            prefix + "position_y": p["position"]["y"],
            prefix + "shield_strength": p["shield_strength"],
            prefix + "stock": p["stock"]
        })
        
    return flat

def process_gamestate(gamestate):
    """Convert a gamestate object into a structured dictionary."""
    if gamestate.menu_state != melee.enums.Menu.IN_GAME:
        return None

    frame_data = {
        "frame": gamestate.frame,
        "distance": gamestate.distance,
        "stage": gamestate.stage.value if gamestate.stage else None,
        "players": {},
    }

    for port, player in gamestate.players.items():
        if player is None:
            continue
        frame_data["players"][port] = {
            "action": player.action.value if player.action else None,
            "action_frame": player.action_frame,
            "character": player.character.value if player.character else None,
            "controller_state": {
                "button": player.controller_state.button,
                "c_stick_x": player.controller_state.c_stick[0],
                "c_stick_y": player.controller_state.c_stick[1],
                "l_shoulder": player.controller_state.l_shoulder,
                "r_shoulder": player.controller_state.r_shoulder,
                "main_stick_x": player.controller_state.main_stick[0],
                "main_stick_y": player.controller_state.main_stick[1]
            },
            "facing": player.facing,
            "hitlag_left": player.hitlag_left,
            "hitstun_frames_left": player.hitstun_frames_left,
            "jumps_left": player.jumps_left,
            "off_stage": player.off_stage,
            "on_ground": player.on_ground,
            "percent": player.percent,
            "position": {
                "x": player.position.x if player.position else None,
                "y": player.position.y if player.position else None
            },
            "shield_strength": player.shield_strength,
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
    # print("Processed slp file shape:", data.shape)
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
    """Create input and target sequences from data using NumPy operations.
    
    Args:
        data: numpy array of processed game data of shape (timesteps, features)
        input_len: length of input sequences
        target_len: length of target sequences
    
    Returns:
        tuple: (input_sequences, target_sequences) where each sequence has shape
        (num_sequences, sequence_length, features_dim)
    """
    total_len = input_len + target_len
    n_sequences = len(data) - total_len + 1
    
    # Create sequences using the first axis only
    input_seqs = np.lib.stride_tricks.sliding_window_view(data, input_len, axis=0)[:n_sequences]
    target_seqs = np.lib.stride_tricks.sliding_window_view(data, target_len, axis=0)[input_len:input_len + n_sequences]
    
    # Transpose from (num_sequences, num_features, seq_len) to (num_sequences, seq_len, num_features)
    input_seqs = np.transpose(input_seqs, (0, 2, 1))
    target_seqs = np.transpose(target_seqs, (0, 2, 1))
    
    return input_seqs, target_seqs

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
    # print(f"Input shape: {input_seqs.shape}")
    # print(f"Target shape: {target_seqs.shape}")

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

def get_slp_files_from_archive(archive_path: Path) -> List[str]:
    """Get list of .slp files from archive without extracting."""
    if archive_path.suffix.lower() == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as archive:
            return [f for f in archive.namelist() 
                   if f.endswith('.slp') and not f.startswith('__MACOSX')]
    else:  # .7z
        with py7zr.SevenZipFile(archive_path, 'r') as archive:
            return [f for f in archive.getnames() 
                   if f.endswith('.slp')]

def process_archive_files_streaming(archive_path: Union[str, Path], 
                                  output_dir: str, 
                                  input_len: int = 10, 
                                  target_len: int = 5) -> None:
    """Process multiple .slp files from a zip or 7z archive one at a time.
    
    Args:
        archive_path: path to the zip/7z file containing .slp files
        output_dir: directory to save the processed sequences
        input_len: length of input sequences
        target_len: length of target sequences
    """
    archive_path = Path(archive_path)
    archive_type = archive_path.suffix.lower()

    if archive_type not in ['.zip', '.7z']:
        raise ValueError(f"Unsupported archive type: {archive_type}. Must be .zip or .7z")

    processed_count = 0
    failed_count = 0
    failed_files = []

    # Get list of .slp files first
    slp_files = get_slp_files_from_archive(archive_path)
    
    if archive_type == '.zip':
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            with zipfile.ZipFile(archive_path, 'r') as archive:
                for slp_file in tqdm(slp_files, desc="Processing ZIP files"):
                    try:
                        # Extract single file to temp directory
                        archive.extract(slp_file, temp_dir)
                        temp_file_path = temp_dir_path / slp_file
                        base_filename = temp_file_path.stem
                        
                        # Process the file
                        sampled_data, columns = process_slp_file(str(temp_file_path))
                        process_and_save_sequences(
                            sampled_data,
                            output_dir,
                            base_filename,
                            input_len=input_len,
                            target_len=target_len
                        )
                        processed_count += 1
                        
                        # Clean up extracted file
                        temp_file_path.unlink()
                    except Exception as e:
                        print(f"Error processing {slp_file}: {str(e)}")
                        failed_count += 1
                        failed_files.append(slp_file)
    
    else:  # .7z
        for slp_file in tqdm(slp_files, desc="Processing 7Z files"):
            base_filename = Path(slp_file).stem
            
            try:
                with py7zr.SevenZipFile(archive_path, mode='r') as archive:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        archive.extract(path=temp_dir, targets=[slp_file])
                        extracted_path = os.path.join(temp_dir, slp_file)
                        
                        if os.path.exists(extracted_path):
                            sampled_data, columns = process_slp_file(extracted_path)
                            process_and_save_sequences(
                                sampled_data,
                                output_dir,
                                base_filename,
                                input_len=input_len,
                                target_len=target_len
                            )
                            processed_count += 1
            except Exception as e:
                print(f"❌ Failed to process {slp_file}: {e}")
                failed_count += 1
                failed_files.append(slp_file)

    print(f"\nProcessed {processed_count} files from {archive_path}")
    print(f"Failed to process {failed_count} files")
    if failed_files:
        print("Failed files:")
        for f in failed_files:
            print(f"  - {f}")
    print(f"Results saved to: {output_dir}")

def batch_process_archives(archive_dir: Union[str, Path], 
                         output_dir: str, 
                         input_len: int = 10, 
                         target_len: int = 5) -> None:
    """Process all zip and 7z archives in a directory.
    
    Args:
        archive_dir: directory containing zip/7z archives
        output_dir: directory to save the processed sequences
        input_len: length of input sequences
        target_len: length of target sequences
    """
    archive_dir = Path(archive_dir)
    archives = list(archive_dir.glob("*.zip")) + list(archive_dir.glob("*.7z"))
    
    if not archives:
        print(f"No .zip or .7z files found in {archive_dir}")
        return
    
    print(f"Found {len(archives)} archives to process")
    
    for archive_path in archives:
        print(f"\nProcessing archive: {archive_path}")
        try:
            process_archive_files_streaming(
                archive_path,
                output_dir,
                input_len=input_len,
                target_len=target_len
            )
        except Exception as e:
            print(f"Error processing archive {archive_path}: {str(e)}")
            continue

def process_directory_slp_files(
    directory_path: Union[str, Path],
    output_dir: str,
    input_len: int = 10,
    target_len: int = 5
) -> Tuple[int, List[str]]:
    """Process all .slp files in a directory and its subdirectories.
    
    Args:
        directory_path: path to directory containing .slp files
        output_dir: directory to save the processed sequences
        input_len: length of input sequences
        target_len: length of target sequences
    
    Returns:
        Tuple of (number of successfully processed files, list of failed files)
    """
    # Get all .slp files recursively
    slp_files = glob(str(Path(directory_path) / "**/*.slp"), recursive=True)
    
    if not slp_files:
        print(f"No .slp files found in {directory_path}")
        return 0, []

    print(f"Found {len(slp_files)} .slp files to process")
    
    processed_count = 0
    failed_files = []

    # Process each file
    for slp_path in tqdm(slp_files, desc="Processing .slp files"):
        try:
            base_filename = Path(slp_path).stem
            sampled_data, columns = process_slp_file(slp_path)
            process_and_save_sequences(
                sampled_data,
                output_dir,
                base_filename,
                input_len=input_len,
                target_len=target_len
            )
            processed_count += 1
        except Exception as e:
            failed_files.append(f"{slp_path}: {str(e)}")

    # Print summary
    print(f"\nSuccessfully processed {processed_count} files")
    if failed_files:
        print(f"Failed to process {len(failed_files)} files:")
        for f in failed_files:
            print(f"  - {f}")
    print(f"Results saved to: {output_dir}")

    return processed_count, failed_files

# Example usage:
if __name__ == "__main__":
    slp_file = "./data/Stream-Game_20220828T223051.slp"
    output_dir = "./processed_data"
    base_filename = "example1"
    
    # Process and save in both .npy and .csv formats
    shape = main(slp_file, output_dir, base_filename, save_csv=True)
    print(f"Processed data shape: {shape}")

    # Process a single archive
    archive_path = "./data/matches.zip"  # or "matches.7z"
    process_archive_files_streaming(archive_path, output_dir)

    # Or process multiple archives in a directory
    archive_dir = "./data/archives"
    batch_process_archives(archive_dir, output_dir)

    # Process all .slp files in a directory
    slp_directory = "./data"
    processed_count, failed = process_directory_slp_files(slp_directory, output_dir)
    print(f"Processed {processed_count} files with {len(failed)} failures")
