# icl_smash
Smash bros. bot using ICL. 

Note that the libmelee folder is from the dev branch of vladfi1's fork of libmelee (https://github.com/vladfi1/libmelee/tree/dev), a library used to interact with the Dolphin/Slippi emulator. If you plan on cloning the repository, make sure to recursively clone submodules as this counts as one.

`run_game.py` is used to run the emulator and sets our trained model against a CPU level 1. Note that to run it on the emulator provided in this zip file, you will need a Melee ISO which is obtained legally by grabbing it from your own Super Smash Bros. Melee (for Gamecube) disk.

`data_processing.py` has all the utility functions involving processing raw `.slp` (replay files) from games.

`data_loading.py` will use the processed data and load it in a usable format for the model in `models.py` to use. 

We have saved a model in `models_params`, and the various `.ipynb` notebooks involve EDA and experimenting with training the model/running the model. 

`live_inference.py` contains the logic of keeping a context window to input into the model at each timestep, along with based on a models prediction from the gamestate input, making our bot actually perform actions in the emulator.


Additionally, please view our processed data at this gdrive link: https://drive.google.com/drive/folders/1HPt5PipV2PQQLsrA4yJqLrI_EZaCumhG?dmr=1&ec=wgc-drive-hero-goto under the folder `processed_data`.  Importantly, the `slp.ipynb` in this google drive above contains our updated code training.

