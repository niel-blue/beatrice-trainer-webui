# lang_en.py
lang_data = {
    "title": "Beatrice-Trainer Unofficial Simple WebUI (for 2.0.0-rc.0)",

    "input_folder": "[ 1. Dataset Folder ]",
    "input_folder_place": "Path to the folder containing the training dataset (speaker folders)",
    "input_folder_alert1": "Dataset folder path is not provided",
    "input_folder_alert2": "Dataset folder does not exist or the path is incorrect",
    "input_folder_info": "Specify the path to the parent folder containing speaker folders (e.g., speaker01).\nExample: If speaker01, speaker02, etc., are in C:\\data\\datafolder, enter C:\\data\\datafolder.",

    "output_folder": "[ 2. Output Folder ]",
    "output_folder_place": "Path to the folder where trained models and configuration files will be saved",
    "output_folder_alert1": "Output folder path is not provided",
    "output_folder_alert2": "Output folder path is invalid",
    "output_folder_info": "The folder where training results (checkpoints, configuration files, etc.) will be saved. It will be created automatically if it does not exist.\nFor additional training, specify the folder containing the target checkpoint.",

    "checkpoint": "[ 3. (Optional) Checkpoint for Additional Training ]",
    "checkpoint_place": "Checkpoint filename for resuming or additional training (e.g., checkpoint_xxxx_00010000.pt.gz)",
    "checkpoint_alert": "The specified checkpoint file does not exist or the filename is incorrect",
    "checkpoint_info": "Specify this for resuming interrupted training or for additional training on an existing model. The file must be located in the output folder.",

    "args_alert": "Some fields are not filled",
    "config_save_info": "Saving config.json with the current settings...",

    "backup_info": "Backing up existing {path}...",
    "rename_info": "Renaming {src} to {dest} (file for resuming training)...",
    "train_start": "Starting new training",
    "addtrain_start": "Resuming training from the specified checkpoint",

    "reset": "Reset Settings",
    "tensorboard": "Launch TensorBoard",
    "tensorboard_alert": "Output folder is not specified",
    "train": "Start Training",

    "n_steps_info": "Total training steps. Batch size × steps estimates training amount and impacts quality.",
    "batch_size_info": "The number of data samples processed at once. Larger values consume more VRAM but can stabilize and speed up training.",
    "num_workers_info": "The number of threads for parallel data loading. Matching the CPU core count improves efficiency but increases memory usage.",
    "save_interval_info": "Saves a checkpoint file every specified number of steps.\nTypically set to the same value as the [Evaluation Interval].",
    "evaluation_interval_info": "Performs evaluation (generating test audio) every specified number of steps to monitor training progress.\nTypically set to the same value as the [Save Interval].",

    "in_sample_rate_info": "Input audio sampling rate (fixed value)",
    "out_sample_rate_info": "Output audio sampling rate (fixed value)",
    "record_metrics_info": "Records detailed loss and metrics to TensorBoard. Disabling this slightly reduces processing load but prevents training analysis.",

    # Performance and Debugging Descriptions
    "use_amp_info": "Use AMP (Automatic Mixed Precision) to reduce VRAM usage and speed up training by employing half-precision calculations. However, it may cause instability in some environments.",
    "san_info": "Use a SAN (Self-Attention Network) based discriminator. It improves the quality of generated audio discrimination but increases computational cost.",
    "profile_info": "Records detailed training process information for bottleneck analysis and debugging. Enabling this reduces processing speed.",

    # WebUI Headings
    "basic_training": "Basic Training Settings",
    "advanced_options": "Advanced Settings",
    "learning_rate_optimizer": "Learning Rate / Optimizer",
    "loss_weights": "Loss Weights",
    "augmentation_options": "Data Augmentation Options",
    "audio_model": "Audio / Model",
    "file_paths": "File Paths",
    "performance_debug": "Performance / Debug"
}