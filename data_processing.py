import os
import melee
import numpy as np
import pandas as pd

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

    # Process player data
    for port in sorted(frame["players"].keys()):
        p = frame["players"][port]
        prefix = f"p{port}_"

        flat.update({
            prefix + "action": p["action"],
            prefix + "action_frame": p["action_frame"],
            prefix + "character": p["character"],
            prefix + "character_selected": p["character_selected"],
            prefix + "controller_button": p["controller_state"]["button"],
            prefix + "controller_c_stick_x": p["controller_state"]["c_stick_x"],
            prefix + "controller_c_stick_y": p["controller_state"]["c_stick_y"],
            prefix + "controller_l_shoulder": p["controller_state"]["l_shoulder"],
            prefix + "controller_r_shoulder": p["controller_state"]["r_shoulder"],
            prefix + "controller_main_stick_x": p["controller_state"]["main_stick_x"],
            prefix + "controller_main_stick_y": p["controller_state"]["main_stick_y"],
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
                prefix + "subtype": proj["subtype"],
                prefix + "type": proj["type"]
            })
        else:
            for key in ["frame", "owner", "position_x", "position_y", "speed_x", "speed_y", "subtype", "type"]:
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
            "character_selected": player.character_selected,
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
    np_array = df.to_numpy()
    
    # Sample the data
    sampled_idxs = np.arange(0, np_array.shape[0], sample_freq)
    sampled_data = np_array[sampled_idxs]
    
    return sampled_data, df.columns.tolist()

def save_processed_data(sampled_data, columns, output_dir, base_filename):
    """Save processed data and column names."""
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, f"{base_filename}_data.npy"), sampled_data)
    # np.save(os.path.join(output_dir, f"{base_filename}_columns.npy"), np.array(columns)) # Column Names

def main(slp_file_path, output_dir, base_filename, sample_freq=15):
    """Main function to process a SLP file and save the results."""
    sampled_data, columns = process_slp_file(slp_file_path, sample_freq)
    save_processed_data(sampled_data, columns, output_dir, base_filename)
    return sampled_data.shape

# Example usage:
if __name__ == "__main__":
    slp_file = "./data/Stream-Game_20220828T223051.slp"
    output_dir = "./processed_data"
    base_filename = "example1"
    
    shape = main(slp_file, output_dir, base_filename)
    print(f"Processed data shape: {shape}")