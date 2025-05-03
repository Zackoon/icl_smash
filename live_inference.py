from libmelee.melee.techskill import multishine
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
        
        total_dim = self.continuous_dim + self.num_enums
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

    def _load_model(self, model_path):
        """Load the trained model"""
        # Define model architecture (same as during training)
        enum_dims = {
            'stage': STAGES_VOCAB_AMOUNT,
            'p1_action': ACTIONS_VOCAB_AMOUNT,
            'p1_character': CHARACTERS_VOCAB_AMOUNT,
            'p2_action': ACTIONS_VOCAB_AMOUNT,
            'p2_character': CHARACTERS_VOCAB_AMOUNT
        }
        
        embedding_dims = {
            'stage': 16,
            'p1_action': 64,
            'p1_character': 16,
            'p2_action': 64,
            'p2_character': 16
        }
        
        model = MeleeEncoderDecoder(
            continuous_dim=54,
            enum_dims=enum_dims,
            embedding_dims=embedding_dims,
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
            
            # Ensure std doesn't have zeros to avoid division by zero
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
        data_normalized = np.concatenate((continuous_data_normalized, data[..., -self.num_enums:]), axis=-1)
        # Add to buffer
        self.frame_buffer.append(data_normalized)
        
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
                    num_enums=self.num_enums,
                    all_inputs=buffer_array
                )
            except Exception as e:
                print(f"Error creating dataset: {e}")
                raise
            
            # Get input tensors from dataset
            try:
                src_cont = dataset.inputs.continuous
                src_enum = {
                    'stage': dataset.inputs.enums.stage,
                    'p1_action': dataset.inputs.enums.p1_action,
                    'p1_character': dataset.inputs.enums.p1_character,
                    'p2_action': dataset.inputs.enums.p2_action,
                    'p2_character': dataset.inputs.enums.p2_character
                }
                
                # Debug enum values and clamp to valid ranges
                for name, tensor in src_enum.items():
                    min_val = tensor.min().item()
                    max_val = tensor.max().item()
                    print(f"Enum {name} values: min={min_val}, max={max_val}")
                    
                    # First, ensure all values are non-negative
                    if min_val < 0:
                        print(f"WARNING: {name} has negative values, clamping to non-negative")
                        src_enum[name] = torch.clamp(tensor, 0, None)
                    
                    # Then, check embedding size and clamp to valid range
                    if hasattr(self.model.enum_embedder.embeddings, name):
                        embed_size = self.model.enum_embedder.embeddings[name].weight.shape[0]
                        print(f"Embedding size for {name}: {embed_size}")
                        
                        # Clamp values to valid range
                        if max_val >= embed_size:
                            print(f"WARNING: {name} has values >= embedding size {embed_size}, clamping")
                            src_enum[name] = torch.clamp(src_enum[name], 0, embed_size - 1)
            except Exception as e:
                print(f"Error preparing input tensors: {e}")
                raise
            
            # Initialize decoder input with zeros (just need a single token)
            batch_size = 1 # TODO: Must change once it is switched to predicting the next 5 frames at inference time
            device = src_cont.device
            
            tgt_cont = torch.zeros((batch_size, 5, self.continuous_dim), device=device)
            tgt_enum = {
                name: torch.zeros((batch_size, 5), dtype=torch.long, device=device)
                for name in src_enum.keys()
            }
            
            # Make prediction for just the next frame
            try:
                with torch.no_grad():
                    # Forward pass
                    print("src_cont", src_cont.shape, "tgt_cont", tgt_cont.shape)
                    cont_pred, enum_pred = self.model(src_cont, src_enum, tgt_cont, tgt_enum)
                    
                    # Check for NaN values in predictions
                    if torch.isnan(cont_pred).any():
                        print("Warning: NaN values detected in continuous predictions")
                        # Replace NaNs with zeros
                        cont_pred = torch.nan_to_num(cont_pred, nan=0.0)
                    
                    # Get only the first prediction
                    next_cont = cont_pred[:, 0, :].squeeze(0).numpy()
                    next_enum = {
                        name: torch.argmax(pred[:, 0, :], dim=1).squeeze(0).item()
                        for name, pred in enum_pred.items()
                    }
                    
                    # Denormalize the continuous predictions if we have normalization stats
                    if self.feature_means is not None and self.feature_stds is not None:
                        next_cont = (next_cont * self.feature_stds) + self.feature_means
                        
                        # Check for NaN values after denormalization
                        if np.isnan(next_cont).any():
                            print("Warning: NaN values detected after denormalization")
                            next_cont = np.nan_to_num(next_cont, nan=0.0)
            except Exception as e:
                print(f"Error during model prediction: {e}")
                raise
            
            return {
                'continuous': next_cont,
                'enums': next_enum
            }
        
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

            # TODO: Note that  for buttons, anything > 0 will count as a button press
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
                    main_x = min(max((main_x + 1) / 2, 0), 1)
                    main_y = min(max((main_y + 1) / 2, 0), 1)
                    controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, main_x, main_y)
                    print(f"Tilting main stick: x={main_x:.2f}, y={main_y:.2f}")
                case 'C':
                    c_x, c_y = value
                    # Ensure values are between 0 and 1 for tilt_analog
                    c_x = min(max((c_x + 1) / 2, 0), 1)
                    c_y = min(max((c_y + 1) / 2, 0), 1)
                    controller.tilt_analog(melee.enums.Button.BUTTON_C, c_x, c_y)
                    print(f"Tilting C stick: x={c_x:.2f}, y={c_y:.2f}")
                case _:
                    # Ignore any unknown buttons
                    print(f"Unknown button: {button}")
        
        # Flush the controller to ensure inputs are sent
        # controller.flush()
        
