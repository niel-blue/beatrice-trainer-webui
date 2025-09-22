# VERSION = "25.09.22"
# lang_en.py
lang_data = {
    "LNG_TITLE": "Beatrice-Trainer Unofficial Simple WebUI (for 2.0.0-rc.0)",
    "LNG_TRAIN_DESC": "Set the input/output folders and various parameters, then click the [Add Task to Queue] button to add a task (multiple tasks are possible). Click [Start Training] to execute the tasks in order.",
    
    "LNG_STOP_SUCCESS_MESSAGE": "Training process stopped. Clearing the queue.",
    "LNG_STOP_NO_PROCESS_MESSAGE": "No training process is currently running. Clearing the queue.",
    "LNG_RESUME_WARNING": "The specified n_steps value ({n_steps_val}) is less than or equal to the existing number of steps ({existing_steps}) in config.json. Please specify a larger value to continue training.",
    "LNG_CONFIG_LOAD_ERROR": "An issue occurred while loading config.json: {e}",
    "LNG_CONFIG_NOT_FOUND_WARNING": "config.json for additional training was not found. Handling as new training.",
    "LNG_LATEST_CHECKPOINT_BACKUP": "Backed up the existing latest checkpoint: {backup_path}",
    
    "LNG_NO_AUDIO_FILES_IN_ROOT_ALERT": "Audio files are in the root of the specified folder. Please create subfolders for each speaker and place the audio in them.",
    "LNG_NO_AUDIO_FILES_IN_SUBFOLDERS_ALERT": "No audio files found. Please check if the audio is placed in the speaker folders.",

    # Training Tab
    "LNG_TAB_TRAIN": "Training",

    "LNG_INPUT_FOLDER": "Dataset Folder",
    "LNG_INPUT_FOLDER_INFO": "Create speaker folders inside the data folder and place audio files in them. Example: C:\\data\\datafolder\\speaker01\\001.wav",
    "LNG_INPUT_FOLDER_PLACE": "e.g., If speaker01, speaker02, etc., are in C:\\data\\datafolder, specify the path up to C:\\data\\datafolder.",
    "LNG_INPUT_FOLDER_ALERT": "Dataset folder path is not provided or the path is incorrect",

    "LNG_OUTPUT_FOLDER": "Output Folder",
    "LNG_OUTPUT_FOLDER_PLACE": "For new training, specify any folder path. For resuming or additional training, specify the folder containing the target checkpoint.",
    "LNG_OUTPUT_FOLDER_ALERT": "Output folder path is not provided or the path is incorrect",

    "LNG_CHECKPOINT": "(Optional) Checkpoint for Additional Training",
    "LNG_CHECKPOINT_PLACE": "Specify the filename only when you want to resume training from a file other than checkpoint_latest.pt.gz. Example: checkpoint_100000_00001000.pt.gz",

    "LNG_CONFIG_SAVE_INFO": "config.json has been generated.",
    "LNG_TENSORBOARD_ALERT": "To launch TensorBoard, you must first specify an output folder.",
    
    "LNG_BASIC_TRAINING": "Basic Training Settings",
    
    "LNG_N_STEPS_INFO": "Total number of training steps. Greatly affects quality. Increasing this also increases training time.",
    "LNG_BATCH_SIZE_INFO": "Number of data samples processed at once. Increasing this can improve stability and greatly affect training speed and quality, but also increases VRAM usage.",
    "LNG_NUM_WORKERS_INFO": "Number of threads for parallel data loading. Matching the CPU core count improves efficiency but increases memory consumption.",
    "LNG_SAVE_INTERVAL_INFO": "Saves a checkpoint file every specified number of steps.",
    "LNG_EVALUATION_INTERVAL_INFO": "Performs evaluation (generating test audio) every specified number of steps to monitor training progress.",

    "LNG_ADVANCED_OPTIONS": "Advanced Settings",
    
    "LNG_LEARNING_RATE_OPTIMIZER": "Learning Rate / Optimizer",
    "LNG_LOSS_WEIGHTS": "Loss Weights",
    "LNG_AUGMENTATION_OPTIONS": "Data Augmentation Options",
    "LNG_AUDIO_MODEL": "Audio / Model Parameters",
    "LNG_FILE_PATHS": "File Paths",
    "LNG_PERFORMANCE_DEBUG": "Performance / Debug",

    "LNG_IN_SAMPLE_RATE_INFO": "Cannot be changed.",
    "LNG_OUT_SAMPLE_RATE_INFO": "Cannot be changed.",
    
    "LNG_TASK_MONITOR": "### [Task Monitor]\n",
    "LNG_TASK_MONITOR_EMPTY": "The queue is currently empty. Please add a task.",
    "LNG_QUEUE_TASK_ADDED": "Task added to queue: {output_folder}",
    "LNG_QUEUE_TASK_RESUMED": "Added to queue as additional training: {output_folder}",
    "LNG_QUEUE_TASK_NEW": "Added to queue as new training: {output_folder}",
    "LNG_QUEUE_STATUS_PENDING": "[Pending ◼]",
    "LNG_QUEUE_STATUS_IN_PROGRESS": "[In Progress ▶]",
    "LNG_QUEUE_STATUS_COMPLETED": "[Completed ✅]",
    "LNG_QUEUE_STATUS_ERROR": "[Error ❌]",
    "LNG_QUEUE_STATUS_STOPPED": "[Stopped 🛑]",
    "LNG_QUEUE_START_INFO": "Starting task from queue: {output_folder}",
    "LNG_USER_TERMINATED_MESSAGE": "Task was interrupted by the user.",
    "LNG_ALL_TASKS_COMPLETED_MESSAGE": "All tasks in the queue have been completed.",

    "LNG_ADD_TASK_BUTTON": "Add Task to Queue",
    "LNG_START_TRAINING_BUTTON": "Start Training",
    "LNG_STOP_TRAINING_BUTTON": "Stop / Clear Queue",
    "LNG_TENSORBOARD_BUTTON": "Launch TensorBoard",

    # Dataset Preprocessing Tab
    "LNG_TAB_DATASET_PROCESSING": "Audio File Splitter",
    "LNG_DATASET_PROCESSING_DESC": "Splits audio files (wav, ogg, mp3, flac, etc.) into specified lengths and converts them to wav or flac format.",
    
    "LNG_INPUT_DIR_PREP_LABEL": "Input Folder",
    "LNG_INPUT_DIR_PREP_PLACE": "Specify the folder path containing the audio files you want to preprocess (multiple files possible).",
    "LNG_OUTPUT_DIR_PREP_LABEL": "Output Folder (Optional)",
    "LNG_OUTPUT_DIR_PREP_PLACE": "Specify the save location for the processed audio files. If left blank, a new folder will be automatically created inside the input folder for output.",
    "LNG_SEGMENT_DURATION_LABEL": "Audio Segment Length (seconds)",
    "LNG_SEGMENT_DURATION_INFO": "Splits audio into segments of the specified length. (*4 seconds recommended)",
    "LNG_OUTPUT_SAMPLERATE_LABEL": "Sampling Rate",
    "LNG_OUTPUT_SAMPLERATE_INFO": "Select the sampling rate for the output audio files. (*16000Hz recommended)",
    "LNG_OUTPUT_FORMAT_LABEL": "Select Output Format",
    "LNG_OUTPUT_FORMAT_INFO": "Choose the file format to save.",
    "LNG_SILENCE_REMOVAL_OPTIONS": "Silence Removal Options",
    "LNG_SILENCE_REMOVAL_OPTIONS_INFO": "Controls whether to automatically remove silent parts.",
    "LNG_SILENCE_THRESHOLD_LABEL": "Silence Threshold (dBFS)",
    "LNG_SILENCE_THRESHOLD_INFO": "Considers any volume below this value as silence.",
    "LNG_MIN_SILENCE_DURATION_LABEL": "Minimum Silence Duration (milliseconds)",
    "LNG_MIN_SILENCE_DURATION_INFO": "Removes silence segments that are longer than this duration.",

    "LNG_STATUS_WAITING": "Status: Waiting",
    "LNG_STATUS_SLICE": "Status: Processing",
    "LNG_ERROR_NO_INPUT_FOLDER": "Please specify an input folder.",
    "LNG_DATASET_SLICE": "Use full dataset slicing",    
    "LNG_DATASET_SLICE_INFO": "By specifying the path to an entire dataset that includes multiple speaker folders, it will slice the files while maintaining the folder structure.",
    "LNG_WARNING_NO_AUDIO_FILES": "No audio files were found in the specified folder.",
    "LNG_ERROR_OUTPUT_FOLDER_CREATION": "Failed to create output folder '{output_dir}'<br>{e}",
    "LNG_COMPLETE_WITH_FAILURES": "Processing complete.Success: {processed_count} Failures: {len(failed_files)}.<br>Failed files:<br>{failed_message}",
    "LNG_COMPLETE_SUCCESS": "Processing complete. Processed {processed_count} files. Skipped: {skipped_count}. Output destination: {output_dir}",
    "LNG_SPLIT_BUTTON": "Export Audio Files",
    "LNG_STOP_SLICE_BUTTON": "Stop Processing",
    "LNG_STATUS_STOPPED": "Stopped by user."
}