import torch
import numpy as np
import pandas as pd
from collections import deque
import melee
import os
from models import MeleeEncoderDecoder
from data_loading import INFERENCE_MODE, MeleeDataset, NUM_ENUMS
from data_processing import process_gamestate, flatten_gamestate

ACTIONS_VOCAB_AMOUNT = 404
CHARACTERS_VOCAB_AMOUNT = 35
STAGES_VOCAB_AMOUNT = 10

class LiveGameStatePredictor:
    def __init__(self, model_path, window_size=10, sample_rate=15, stats_dir="./model_params"):
        """
        Initialize the predictor for live inference (next frame only)
        
        Args:
            model_path: Path to the saved model
            window_size: Number of frames to use as context
            stats_dir: Directory containing normalization statistics
        """
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        self.continuous_dim = 54 # TODO: May change with future data

        # Configuration
        self.window_size = window_size
        self.num_enums = NUM_ENUMS
        
        # Single buffer for storing recent frames
        self.frame_buffer = deque(maxlen=window_size * sample_rate)
        
        total_dim = self.continuous_dim # + self.num_enums
        empty_frame = np.zeros((1, 1, total_dim))
        
        # Fill the buffer with empty frames so the model can begin
        for _ in range(window_size * sample_rate):
            self.frame_buffer.append(empty_frame)

        self.sample_idxs = [i for i in range(0, window_size * sample_rate, 15)]

        # Load feature normalization parameters
        self._load_normalization_stats(stats_dir)
        
        # Flag to track if we have enough frames
        self.buffer_ready = False
        
        # Number of continuous features
        self.continuous_dim = 54  # Adjust based on your feature dimension
        
        # Add tracking for previous predictions and frames
        self.previous_prediction = None
        self.consecutive_same_predictions = 0
        self.previous_sampled_frames = None
        self.consecutive_same_frames = 0

    def _load_model(self, model_path):
        """Load the trained model"""
        # Define model architecture (same as during training)
        # enum_dims = {
        #     'stage': STAGES_VOCAB_AMOUNT,
        #     'p1_action': ACTIONS_VOCAB_AMOUNT,
        #     'p1_character': CHARACTERS_VOCAB_AMOUNT,
        #     'p2_action': ACTIONS_VOCAB_AMOUNT,
        #     'p2_character': CHARACTERS_VOCAB_AMOUNT
        # }
        
        # embedding_dims = {
        #     'stage': 16,
        #     'p1_action': 64,
        #     'p1_character': 16,
        #     'p2_action': 64,
        #     'p2_character': 16
        # }
        
        model = MeleeEncoderDecoder(
            continuous_dim=54,
            # enum_dims=enum_dims,
            # embedding_dims=embedding_dims,
            d_model=128,
            nhead=4,
            num_layers=3
        )
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        return model
    
    def _load_normalization_stats(self, stats_dir):
        """
        Load normalization statistics from the model_params directory
        
        Args:
            stats_dir: Directory containing training_mean.pt and training_std.pt files
        """
        mean_path = os.path.join(stats_dir, "training_mean.pt")
        std_path = os.path.join(stats_dir, "training_std.pt")
        
        if os.path.exists(mean_path) and os.path.exists(std_path):
            self.feature_means = torch.load(mean_path, map_location=torch.device('cpu')).numpy()
            self.feature_stds = torch.load(std_path, map_location=torch.device('cpu')).numpy()
            print(f"Loaded normalization statistics from {stats_dir}")
            
            # Check for NaN values in loaded statistics
            if np.isnan(self.feature_means).any():
                nan_count = np.isnan(self.feature_means).sum()
                print(f"WARNING: {nan_count} NaN values found in feature_means")
                # Replace NaNs with zeros
                self.feature_means = np.nan_to_num(self.feature_means, nan=0.0)
            
            if np.isnan(self.feature_stds).any():
                nan_count = np.isnan(self.feature_stds).sum()
                print(f"WARNING: {nan_count} NaN values found in feature_stds")
                # Replace NaNs with ones (neutral for division)
                self.feature_stds = np.nan_to_num(self.feature_stds, nan=1.0)
            
            # Ensure std doesn't have zeros to avoid division by zero
            zero_count = (self.feature_stds == 0).sum()
            if zero_count > 0:
                print(f"WARNING: {zero_count} zero values found in feature_stds")
            self.feature_stds = np.maximum(self.feature_stds, 1e-6)
            
        else:
            print(f"Warning: Normalization statistics not found in {stats_dir}")
            self.feature_means = None
            self.feature_stds = None

    def normalize_features(self, features):
        """
        Normalize continuous features using pre-computed statistics
        
        Args:
            features: Tensor or numpy array of continuous features
        
        Returns:
            Normalized features (same type as input)
        """

        if self.feature_means is None or self.feature_stds is None:
            print("Warning: Normalization statistics not available, using raw features")
            return features
        
        # Check if input is a PyTorch tensor
        is_tensor = isinstance(features, torch.Tensor)
        
        if is_tensor:
            # For PyTorch tensors
            # Convert to float if needed
            if not torch.is_floating_point(features):
                print(f"Converting tensor from {features.dtype} to float32")
                features = features.float()
            
            # Check for NaN values
            if torch.isnan(features).any():
                print("Warning: NaN values detected in input tensor before normalization")
                features = torch.nan_to_num(features, nan=0.0)
            
            # Ensure feature_means and feature_stds are tensors on the same device
            device = features.device
            feature_means = torch.tensor(self.feature_means, device=device, dtype=features.dtype)
            feature_stds = torch.tensor(self.feature_stds, device=device, dtype=features.dtype)
            
            # Normalize
            normalized = (features - feature_means) / feature_stds
            
            # Check for NaN values after normalization
            if torch.isnan(normalized).any():
                print("Warning: NaN values detected after normalization")
                normalized = torch.nan_to_num(normalized, nan=0.0)
        else:
            # For numpy arrays
            # Convert to float if needed
            if not np.issubdtype(features.dtype, np.floating):
                print(f"Converting numpy array from {features.dtype} to float32")
                features = features.astype(np.float32)
            
            # Check for NaN values
            if np.isnan(features).any():
                print("Warning: NaN values detected in input features before normalization")
                features = np.nan_to_num(features, nan=0.0)
            
            # Normalize
            normalized = (features - self.feature_means) / self.feature_stds
            
            # Check for NaN values after normalization
            if np.isnan(normalized).any():
                print("Warning: NaN values detected after normalization")
                normalized = np.nan_to_num(normalized, nan=0.0)
        
        return normalized
    
    def process_gamestate(self, gamestate):
        """
        Process a new gamestate frame and update buffer
        
        Args:
            gamestate: A melee.GameState object from libmelee
        
        Returns:
            ready: Boolean indicating if we have enough frames to make predictions
        """
        # Extract features from gamestate and normalize them
        flattened_frame = [flatten_gamestate(process_gamestate(gamestate))]
        df = pd.DataFrame(flattened_frame)
        data = df.to_numpy()
        data = np.expand_dims(data, axis=0)
        continuous_data_normalized = self.normalize_features(data[..., :-self.num_enums])
        # print(f"Continuous data normalized shape: {continuous_data_normalized.shape}")
        # data_normalized = np.concatenate((continuous_data_normalized, data[..., -self.num_enums:]), axis=-1)
        # Add to buffer
        self.frame_buffer.append(continuous_data_normalized)
        
        # Check if we have enough frames
        if len(self.frame_buffer) > self.sample_idxs[-1]:
            self.buffer_ready = True
            
        return self.buffer_ready

    def predict_next_frame(self):
        """
        Predict only the next frame based on current buffer state
        
        Returns:
            next_frame: Dictionary with continuous and enum predictions for the next frame
            None if buffer is not ready
        
        Raises:
            Exception: If an error occurs during prediction
        """
        try:
            if not self.buffer_ready:
                print("Buffer not ready yet")
                return None
            
            # Ensure we have enough frames
            if len(self.frame_buffer) < self.window_size:
                print(f"Not enough frames in buffer: {len(self.frame_buffer)} < {self.window_size}")
                return None
            
            # Get evenly spaced frames
            sampled_frames = [self.frame_buffer[i] for i in self.sample_idxs]
            
            # Check if sampled frames are identical to previous ones
            if self.previous_sampled_frames is not None:
                frames_identical = True
                
                if len(sampled_frames) != len(self.previous_sampled_frames):
                    frames_identical = False
                else:
                    for i, (curr_frame, prev_frame) in enumerate(zip(sampled_frames, self.previous_sampled_frames)):
                        # Compare the frames, treating NaN values as equal
                        if not np.all(np.isclose(curr_frame, prev_frame, equal_nan=True)):
                            frames_identical = False
                            # Calculate absolute differences
                            abs_diff = np.abs(curr_frame - prev_frame)
                            # Get max difference and its location
                            max_diff = np.nanmax(abs_diff)
                            max_diff_idx = np.unravel_index(np.nanargmax(abs_diff), abs_diff.shape)
                            # Get mean difference (excluding NaNs)
                            mean_diff = np.nanmean(abs_diff)
                            print(f"Frame {i} changed:")
                            print(f"  Max difference: {max_diff:.6f} at index {max_diff_idx}")
                            print(f"  Mean difference: {mean_diff:.6f}")
                            # Show the actual values at the location of max difference
                            print(f"  Values at max diff location:")
                            print(f"    Current: {curr_frame[max_diff_idx]:.6f}")
                            print(f"    Previous: {prev_frame[max_diff_idx]:.6f}")
                            break
                
                if frames_identical:
                    self.consecutive_same_frames += 1
                    print(f"WARNING: Same sampled frames {self.consecutive_same_frames} times in a row")
                else:
                    self.consecutive_same_frames = 0
                    print("Sampled frames changed!")
            
            # Store current frames for next comparison
            self.previous_sampled_frames = [frame.copy() for frame in sampled_frames]
            
            # Debug info
            print(f"Sampled {len(sampled_frames)} frames")
            
            # Stack frames into a single array
            try:
                buffer_array = np.concatenate(sampled_frames, axis=1)
                print(f"Buffer array shape: {buffer_array.shape}")
            except Exception as e:
                print(f"Error stacking frames: {e}")
                print(f"First frame shape: {sampled_frames[0].shape if sampled_frames else 'No frames'}")
                raise
            
            # Create dataset from buffer using MeleeDataset
            try:
                dataset = MeleeDataset(
                    data_path=INFERENCE_MODE,
                    match_id=1,
                    # num_enums= self.num_enums,
                    all_inputs=buffer_array
                )
            except Exception as e:
                print(f"Error creating dataset: {e}")
                raise
            
            # Get input tensors from dataset
            try:
                src_cont = dataset.inputs.continuous
                # src_enum = {
                #     'stage': dataset.inputs.enums.stage,
                #     'p1_action': dataset.inputs.enums.p1_action,
                #     'p1_character': dataset.inputs.enums.p1_character,
                #     'p2_action': dataset.inputs.enums.p2_action,
                #     'p2_character': dataset.inputs.enums.p2_character
                # }
                
                # Debug enum values and clamp to valid ranges
                # for name, tensor in src_enum.items():
                #     min_val = tensor.min().item()
                #     max_val = tensor.max().item()
                #     print(f"Enum {name} values: min={min_val}, max={max_val}")
                    
                #     # First, ensure all values are non-negative
                #     if min_val < 0:
                #         print(f"WARNING: {name} has negative values, clamping to non-negative")
                #         #src_enum[name] = torch.clamp(tensor, 0, None)
                    
                #     # Then, check embedding size and clamp to valid range
                #     if hasattr(self.model.enum_embedder.embeddings, name):
                #         embed_size = self.model.enum_embedder.embeddings[name].weight.shape[0]
                #         print(f"Embedding size for {name}: {embed_size}")
                        
                #         # Clamp values to valid range
                #         if max_val >= embed_size:
                #             print(f"WARNING: {name} has values >= embedding size {embed_size}, clamping")
                #             src_enum[name] = torch.clamp(src_enum[name], 0, embed_size - 1)
            except Exception as e:
                print(f"Error preparing input tensors: {e}")
                raise
            
            # Initialize decoder input with zeros (just need a single token)
            batch_size = 1 # TODO: Must change once it is switched to predicting the next 5 frames at inference time
            device = src_cont.device
            
            tgt_cont = torch.zeros((batch_size, 5, self.continuous_dim), device=device)
            # tgt_enum = {
            #     name: torch.zeros((batch_size, 5), dtype=torch.long, device=device)
            #     for name in src_enum.keys()
            # }
            
            # Make prediction for just the next frame
            try:
                with torch.no_grad():
                    # Forward pass
                    print("src_cont", src_cont.shape, "tgt_cont", tgt_cont.shape)
                    
                    # Check for NaNs in input tensors
                    if torch.isnan(src_cont).any():
                        print("WARNING: NaN values detected in src_cont input tensor")
                        # Find which elements are NaN
                        nan_indices = torch.where(torch.isnan(src_cont))
                        print(f"NaN indices in src_cont: {list(zip(*[idx.tolist() for idx in nan_indices]))[:10]} (showing first 10)")
                        # Replace NaNs with zeros
                        src_cont = torch.nan_to_num(src_cont, nan=0.0)
                    
                    # Check enum inputs for invalid values
                    # for name, tensor in src_enum.items():
                    #     if torch.isnan(tensor.float()).any():
                    #         print(f"WARNING: NaN values detected in {name} enum tensor")
                    #         src_enum[name] = torch.nan_to_num(tensor.float(), nan=0.0).long()
                    
                    # Forward pass
                    cont_pred = self.model(src_cont, tgt_cont)
                    
                    # Check for NaNs in continuous predictions
                    if torch.isnan(cont_pred).any():
                        print("WARNING: NaN values detected in continuous predictions")
                        # Find which elements are NaN
                        nan_indices = torch.where(torch.isnan(cont_pred))
                        print(f"NaN indices in cont_pred: {list(zip(*[idx.tolist() for idx in nan_indices]))[:10]} (showing first 10)")
                        # Count NaNs
                        nan_count = torch.isnan(cont_pred).sum().item()
                        total_elements = cont_pred.numel()
                        print(f"NaN count: {nan_count}/{total_elements} ({nan_count/total_elements*100:.2f}%)")
                        # Replace NaNs with zeros
                        cont_pred = torch.nan_to_num(cont_pred, nan=0.0)
                    
                    # Check for NaNs in enum predictions
                    # for name, pred in enum_pred.items():
                    #     if torch.isnan(pred).any():
                    #         print(f"WARNING: NaN values detected in {name} enum predictions")
                    #         enum_pred[name] = torch.nan_to_num(pred, nan=0.0)
                    
                    # Get only the first prediction
                    next_cont = cont_pred[:, 0, :].squeeze(0).numpy()
                    # next_enum = {
                    #     name: torch.argmax(pred[:, 0, :], dim=1).squeeze(0).item()
                    #     for name, pred in enum_pred.items()
                    # }
                    
                    # Check for NaNs in numpy array
                    if np.isnan(next_cont).any():
                        print("WARNING: NaN values detected in next_cont numpy array")
                        # Find which elements are NaN
                        nan_indices = np.where(np.isnan(next_cont))[0]
                        print(f"NaN indices in next_cont: {nan_indices[:10]} (showing first 10)")
                        # Count NaNs
                        nan_count = np.isnan(next_cont).sum()
                        total_elements = next_cont.size
                        print(f"NaN count: {nan_count}/{total_elements} ({nan_count/total_elements*100:.2f}%)")
                        # Replace NaNs with zeros
                        next_cont = np.nan_to_num(next_cont, nan=0.0)
                    
                    # Denormalize the continuous predictions if we have normalization stats
                    if self.feature_means is not None and self.feature_stds is not None:
                        next_cont = (next_cont * self.feature_stds) + self.feature_means
                        
                        # Check for NaN values after denormalization
                        if np.isnan(next_cont).any():
                            print("WARNING: NaN values detected after denormalization")
                            # Find which elements are NaN
                            nan_indices = np.where(np.isnan(next_cont))[0]
                            print(f"NaN indices after denormalization: {nan_indices[:10]} (showing first 10)")
                            # Check if NaNs are coming from feature_means or feature_stds
                            if np.isnan(self.feature_means).any():
                                print("WARNING: NaN values detected in feature_means")
                                nan_indices = np.where(np.isnan(self.feature_means))[0]
                                print(f"NaN indices in feature_means: {nan_indices[:10]} (showing first 10)")
                            
                            if np.isnan(self.feature_stds).any():
                                print("WARNING: NaN values detected in feature_stds")
                                nan_indices = np.where(np.isnan(self.feature_stds))[0]
                                print(f"NaN indices in feature_stds: {nan_indices[:10]} (showing first 10)")
                            
                            next_cont = np.nan_to_num(next_cont, nan=0.0)
                    
                    # Final check for NaNs in the return values
                    if np.isnan(next_cont).any():
                        print("WARNING: Final next_cont still contains NaN values")
                        next_cont = np.nan_to_num(next_cont, nan=0.0)
                    
                    return {
                        'continuous': next_cont,
                        # 'enums': next_enum
                    }
            except Exception as e:
                print(f"Error during model prediction: {e}")
                import traceback
                traceback.print_exc()
                raise
        except Exception as e:
            print(f"Error in predict_next_frame: {e}")
            import traceback
            traceback.print_exc()
            raise

    # The bot, by convention will be p1
    def predict_next_frame_p1_buttons(self):
        try:
            next_frame = self.predict_next_frame()
            
            # Check if prediction was successful
            if next_frame is None:
                print("No next frame prediction available")
                return None
            
            # Check for NaN values in continuous features
            if np.isnan(next_frame['continuous']).any():
                print("Warning: NaN values detected in continuous features")
                next_frame['continuous'] = np.nan_to_num(next_frame['continuous'], nan=0.0)
            
            button_to_next_frame_index = {
                'A': 9,
                'B': 10,
                'X': 11,
                'Y': 12,
                'Z': 13,
                'L': 5,  # analog value
                'R': 6,  # analog value
            }
            # 'MAIN': (7, 8),     # (main_stick_x, main_stick_y)
            # 'C': (3, 4),        # (c_stick_x, c_stick_y)

            # TODO: Note that for buttons, anything > 0 will count as a button press. Could consider taking the max 
            button_values =  {key: next_frame['continuous'][0, 0, value] for key, value in button_to_next_frame_index.items() if next_frame['continuous'][0, 0, value] > 0}
            
            # TODO: L and R shield and Z override any other input, so impose a harsher threshold
            if 'L' in button_values and button_values['L'] < 0.5:
                del button_values['L']
            if 'R' in button_values and button_values['R'] < 0.5:
                del button_values['R']
            if 'Z' in button_values and button_values['Z'] < 0.5:
                del button_values['Z']

            # Handle analog stick/c stick separately, as 0 would indicate that the stick is being pushed to the leftmost side
            button_values['MAIN'] = (next_frame['continuous'][0, 0, 7], next_frame['continuous'][0, 0, 8])
            button_values['C'] = (next_frame['continuous'][0, 0, 3], next_frame['continuous'][0,0,4])
            
            return button_values
        except Exception as e:
            print(f"Error in predict_next_frame_p1_buttons: {e}")
            import traceback
            traceback.print_exc()
            raise

    def perform_button_presses(self, button_values, controller, gamestate=None):
        """
        Apply predicted button presses to the controller
        
        Args:
            button_values: Dictionary of button values from predict_next_frame_p1_buttons
            controller: melee.Controller object to control
            gamestate: Optional current game state for more context-aware controls
        """
        
        # Requires some releasing of the controller to actually work
        # Right now, it'll release on the 3rd frame of their action all the time
            # Ideally, this would instead be when non of the inputs are strongly pressed
            # however, our model is just repeating the same output every time
        if gamestate.players[1].action_frame == 3:
            controller.release_all()
            return
        print("Button values:", button_values)
        # Process each button in the prediction
        for button, value in button_values.items():
            match button:
                case 'A':
                    controller.press_button(melee.enums.Button.BUTTON_A)
                    print("Pressing A button")
                case 'B':
                    controller.press_button(melee.enums.Button.BUTTON_B)
                    print("Pressing B button")
                case 'X':
                    controller.press_button(melee.enums.Button.BUTTON_X)
                    print("Pressing X button")
                case 'Y':
                    controller.press_button(melee.enums.Button.BUTTON_Y)
                    print("Pressing Y button")
                case 'Z':
                    controller.press_button(melee.enums.Button.BUTTON_Z)
                    print("Pressing Z button")
                case 'L':
                    # Convert to a value between 0 and 1
                    l_value = min(max(value, 0), 1)
                    controller.press_shoulder(melee.enums.Button.BUTTON_L, l_value)
                    print(f"Pressing L shoulder: {l_value:.2f}")
                case 'R':
                    # Convert to a value between 0 and 1
                    r_value = min(max(value, 0), 1)
                    controller.press_shoulder(melee.enums.Button.BUTTON_R, r_value)
                    print(f"Pressing R shoulder: {r_value:.2f}")
                case 'MAIN':
                    main_x, main_y = value
                    # Ensure values are between 0 and 1 for tilt_analog
                    # main_x = min(max((main_x + 1) / 2, 0), 1)
                    # main_y = min(max((main_y + 1) / 2, 0), 1)
                    controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, main_x, main_y)
                    print(f"Tilting main stick: x={main_x:.2f}, y={main_y:.2f}")
                case 'C':
                    c_x, c_y = value
                    # Ensure values are between 0 and 1 for tilt_analog
                    # c_x = min(max((c_x + 1) / 2, 0), 1)
                    # c_y = min(max((c_y + 1) / 2, 0), 1)
                    controller.tilt_analog(melee.enums.Button.BUTTON_C, c_x, c_y)
                    print(f"Tilting C stick: x={c_x:.2f}, y={c_y:.2f}")
                case _:
                    # Ignore any unknown buttons
                    print(f"Unknown button: {button}")
        
        # Flush the controller to ensure inputs are sent
        # controller.flush()
    # Deprecated since enums removed

    # def test_model_responsiveness(self):
    #     """
    #     Test if the model responds to different inputs
    #     """
    #     print("Testing model responsiveness...")
        
    #     # Create two different test inputs
    #     device = next(self.model.parameters()).device
    #     test_batch_size = 1
    #     test_seq_len = 10
        
    #     # Test input 1 - all zeros
    #     test_cont_1 = torch.zeros((test_batch_size, test_seq_len, self.continuous_dim), device=device)
    #     test_enum_1 = {
    #         name: torch.zeros((test_batch_size, test_seq_len), dtype=torch.long, device=device)
    #         for name in ['stage', 'p1_action', 'p1_character', 'p2_action', 'p2_character']
    #     }
        
    #     # Test input 2 - random values
    #     test_cont_2 = torch.rand((test_batch_size, test_seq_len, self.continuous_dim), device=device)
    #     test_enum_2 = {
    #         name: torch.randint(0, dim-1, (test_batch_size, test_seq_len), dtype=torch.long, device=device)
    #         for name, dim in {'stage': STAGES_VOCAB_AMOUNT, 
    #                          'p1_action': ACTIONS_VOCAB_AMOUNT,
    #                          'p1_character': CHARACTERS_VOCAB_AMOUNT, 
    #                          'p2_action': ACTIONS_VOCAB_AMOUNT,
    #                          'p2_character': CHARACTERS_VOCAB_AMOUNT}.items()
    #     }
        
    #     # Target inputs (same for both tests)
    #     tgt_cont = torch.zeros((test_batch_size, 5, self.continuous_dim), device=device)
    #     tgt_enum = {
    #         name: torch.zeros((test_batch_size, 5), dtype=torch.long, device=device)
    #         for name in test_enum_1.keys()
    #     }
        
    #     # Run predictions
    #     with torch.no_grad():
    #         cont_pred_1, enum_pred_1 = self.model(test_cont_1, test_enum_1, tgt_cont, tgt_enum)
    #         cont_pred_2, enum_pred_2 = self.model(test_cont_2, test_enum_2, tgt_cont, tgt_enum)
        
    #     # Compare outputs
    #     cont_diff = (cont_pred_1 - cont_pred_2).abs().mean().item()
    #     enum_diff = sum((pred_1[:, 0, :] - pred_2[:, 0, :]).abs().mean().item() 
    #                    for name, (pred_1, pred_2) in 
    #                    zip(test_enum_1.keys(), zip(enum_pred_1.values(), enum_pred_2.values()))) / len(test_enum_1)
        
    #     print(f"Continuous prediction difference: {cont_diff:.6f}")
    #     print(f"Enum prediction difference: {enum_diff:.6f}")
        
    #     if cont_diff < 1e-6 and enum_diff < 1e-6:
    #         print("WARNING: Model produces nearly identical outputs for different inputs!")
    #         print("This suggests the model may not be responsive to input changes.")
    #     else:
    #         print("Model produces different outputs for different inputs as expected.")
        
    #     return cont_diff, enum_diff

    # def check_normalization_stats(self):
    #     """
    #     Check normalization statistics for NaN values and other issues
    
    #     Returns:
    #         dict: Dictionary with diagnostic information
    #     """
    #     results = {
    #         "has_means": self.feature_means is not None,
    #         "has_stds": self.feature_stds is not None,
    #         "issues": []
    #     }
        
    #     if self.feature_means is None or self.feature_stds is None:
    #         results["issues"].append("Missing normalization statistics")
    #         return results
        
    #     # Check for NaNs in means
    #     if np.isnan(self.feature_means).any():
    #         nan_count = np.isnan(self.feature_means).sum()
    #         total = self.feature_means.size
    #         results["issues"].append(f"NaN values in means: {nan_count}/{total} ({nan_count/total*100:.2f}%)")
    #         nan_indices = np.where(np.isnan(self.feature_means))[0]
    #         results["mean_nan_indices"] = nan_indices[:10].tolist()  # First 10 indices
        
    #     # Check for NaNs in stds
    #     if np.isnan(self.feature_stds).any():
    #         nan_count = np.isnan(self.feature_stds).sum()
    #         total = self.feature_stds.size
    #         results["issues"].append(f"NaN values in stds: {nan_count}/{total} ({nan_count/total*100:.2f}%)")
    #         nan_indices = np.where(np.isnan(self.feature_stds))[0]
    #         results["std_nan_indices"] = nan_indices[:10].tolist()  # First 10 indices
        
    #     # Check for zeros in stds (can cause division by zero)
    #     if (self.feature_stds == 0).any():
    #         zero_count = (self.feature_stds == 0).sum()
    #         total = self.feature_stds.size
    #         results["issues"].append(f"Zero values in stds: {zero_count}/{total} ({zero_count/total*100:.2f}%)")
    #         zero_indices = np.where(self.feature_stds == 0)[0]
    #         results["std_zero_indices"] = zero_indices[:10].tolist()  # First 10 indices
        
    #     # Check for very small values in stds (can cause numerical instability)
    #     small_threshold = 1e-6
    #     if ((self.feature_stds < small_threshold) & (self.feature_stds > 0)).any():
    #         small_count = ((self.feature_stds < small_threshold) & (self.feature_stds > 0)).sum()
    #         total = self.feature_stds.size
    #         results["issues"].append(f"Very small values in stds: {small_count}/{total} ({small_count/total*100:.2f}%)")
    #         small_indices = np.where((self.feature_stds < small_threshold) & (self.feature_stds > 0))[0]
    #         results["std_small_indices"] = small_indices[:10].tolist()  # First 10 indices
        
    #     # Check for infinite values
    #     if np.isinf(self.feature_means).any():
    #         inf_count = np.isinf(self.feature_means).sum()
    #         results["issues"].append(f"Infinite values in means: {inf_count}")
        
    #     if np.isinf(self.feature_stds).any():
    #         inf_count = np.isinf(self.feature_stds).sum()
    #         results["issues"].append(f"Infinite values in stds: {inf_count}")
        
    #     # Add basic statistics
    #     results["mean_stats"] = {
    #         "min": float(np.nanmin(self.feature_means)),
    #         "max": float(np.nanmax(self.feature_means)),
    #         "avg": float(np.nanmean(self.feature_means))
    #     }
        
    #     results["std_stats"] = {
    #         "min": float(np.nanmin(self.feature_stds)),
    #         "max": float(np.nanmax(self.feature_stds)),
    #         "avg": float(np.nanmean(self.feature_stds))
    #     }
        
    #     return results

