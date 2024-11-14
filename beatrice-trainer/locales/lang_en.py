# lang_en.py
lang_data = {
    "title": "Beatrice-Trainer beta2 Unofficial Simple WebUI",

    "input_folder": "[ Data Folder Path ]",
    "input_folder_place": "Enter the path to the data folder",
    "input_folder_alert1": "The data folder path has not been entered",
    "input_folder_alert2": "The data folder does not exist, or the path is incorrect",

    "output_folder": "[ Output Folder Path ]",
    "output_folder_place": "Enter the path to the output folder",
    "output_folder_alert1": "The output folder path has not been entered",
    "output_folder_alert2": "The output folder does not exist, or the path is incorrect",

    "args_alert": "Some required fields are missing",

    "config_save_info": "Saving config.json with the current settings…",
    "checkpoint": "[ Select Checkpoint ]",
    "checkpoint_place": "Enter the checkpoint file name for additional training",
    "checkpoint_alert": "The checkpoint file does not exist, or the file name is incorrect",

    "backup_info": "Backing up the existing {path}…",
    "rename_info": "Renaming {src} to {dest}…",
    "train_start": "Starting training",
    "addtrain_start": "Resuming training from the previous checkpoint",

    "reset": "Reset",
    "tensorboard": "TensorBoard",
    "tensorboard_alert": "Output folder path not specified",
    "train": "Train",

    "input_folder_info": "Specify the path to the data folder, not the speaker folder containing the WAV files. \nFor example, if your path is \nC:\Beatrice-trainer\datafolder\person01\P1_001.wav,\nenter C:\Beatrice-trainer\datafolder.",
    "output_folder_info": "For new training, specify an empty folder.\nFor resuming or additional training, select the previously used folder.",
    "checkpoint_info": "In the current version, specifying the 'completed training file' for additional training will cause errors after starting. Instead, specify 'the file just before completion,' not the one after the training is finished.\nFor example, if you completed training up to 10000 steps:\ncheckpoint_input_person01_00007000.pt\ncheckpoint_input_person01_00008000.pt <--- Use this one\ncheckpoint_input_person01_00010000.pt\n",
    "batch_size_info": "Specifies the training batch size per step. Increasing this value uses more VRAM.",
    "num_workers_info": "Specifies the number of parallel workers for data transfer during training. Increasing this value uses more main memory.",
    "n_steps_info": "Specifies the total number of training steps. The total training volume is 'batch size x steps.'\nA checkpoint file is saved every 2000 steps",

    "in_sample_rate_info": "Non-adjustable!!!",
    "out_sample_rate_info": "Non-adjustable!!!"
}
