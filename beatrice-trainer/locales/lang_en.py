# lang_en.py
lang_data = {
    "title": "Beatrice-Trainer Unofficial Simple WebUI (for 2.0.0-rc.0)",

    "input_folder": "[ 1. Specify Data Folder ]",
    "input_folder_place": "Path to the folder containing the training dataset (speaker folders)",
    "input_folder_alert1": "Data folder path has not been entered",
    "input_folder_alert2": "The data folder does not exist or the path is incorrect",
    "input_folder_info": "Specify the path to the folder one level above the speaker folders containing the audio files (e.g., speaker01).\nExample: If speaker01, speaker02... are inside C:\\data\\datafolder, enter C:\\data\\datafolder",

    "output_folder": "[ 2. Specify Output Folder ]",
    "output_folder_place": "Path to the folder where the trained model and config file will be output",
    "output_folder_alert1": "Output folder path has not been entered",
    "output_folder_alert2": "The output folder path is invalid",
    "output_folder_info": "This folder saves training results (checkpoints, config files, etc.). It will be created automatically if it does not exist.\nFor additional training, specify the folder containing the checkpoint you want to use.",

    "checkpoint": "[ 3. (Optional) Checkpoint for Additional Training ]",
    "checkpoint_place": "Checkpoint file name for additional training or resuming training (e.g., checkpoint_xxxx_00010000.pt.gz)",
    "checkpoint_alert": "The specified checkpoint file does not exist or the file name is incorrect",
    "checkpoint_info": "Specify this to resume interrupted training or for additional training on an existing model. The file must be in the output folder.\nPlease note that the file format has been changed to the compressed format '.pt.gz'.",

    "args_alert":"Some fields are missing",
    "config_save_info": "Writing config.json with the current settings...",

    "backup_info": "Backing up the existing {path}...",
    "rename_info": "Renaming {src} to {dest} (file for resuming training)...",
    "train_start": "Starting new training",
    "addtrain_start": "Resuming training from the specified checkpoint",

    "reset": "Reset Settings",
    "tensorboard": "Launch TensorBoard",
    "tensorboard_alert": "Output folder is not specified",
    "train": "Start Training",

    "n_steps_info": "Total number of training steps. The training volume is roughly 'batch size x steps.'",
    "batch_size_info": "Amount of data processed at once. A larger size consumes more VRAM but tends to stabilize and speed up training.",
    "num_workers_info": "Number of parallel workers for data loading. Setting this based on your PC's CPU cores can increase efficiency, but it also increases main memory consumption.",
    "save_interval_info": "Saves a checkpoint file every specified number of steps.\nIt is generally recommended to use the same value as evaluation_interval.",
    "evaluation_interval_info": "Performs a validation (generates a test audio) every specified number of steps to check the quality.\nIt is generally recommended to use the same value as save_interval.",

    "in_sample_rate_info": "Input audio sample rate (cannot be changed)",
    "out_sample_rate_info": "Output audio sample rate (cannot be changed)",
    "record_metrics_info": "Whether to record detailed loss graphs to TensorBoard. Turning this off slightly reduces processing load, but prevents detailed analysis of the training status.",
    
    # New keys for WebUI display
    "advanced_options": "Advanced options",
    "learning_rate_optimizer": "Learning Rate / Optimizer",
    "loss_weights": "Loss Weights",
    "augmentation_options": "Augmentation options",
    "audio_model": "Audio / Model",
    "file_paths": "File Paths",
    "performance_debug": "Performance / Debug"
}