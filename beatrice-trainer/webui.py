import gradio as gr
import json
import os
import subprocess
import webbrowser
import locale
import importlib
import shutil
import torch
import torchaudio
from pathlib import Path
import multiprocessing
import threading
import time
import queue
import re

# バージョン情報
VERSION = "25.09.22"

# グローバル変数など
torchaudio.set_audio_backend("sox_io")

training_process = None
is_terminated_by_user = False
training_tasks = []
current_task = None
training_thread = None
current_dir = os.getcwd()
default_config_path = os.path.join(current_dir, "assets", "default_config.json")
is_slicing_terminated_by_user = False # 新しいグローバル変数

# 言語設定ファイルの読み込み
def load_locale():
    try:
        lang_code, _ = locale.getdefaultlocale()
        lang_file = "lang_ja" if lang_code == "ja_JP" else "lang_en"
        lang_module = importlib.import_module(f"locales.{lang_file}")
        lang_data = {key.upper(): value for key, value in lang_module.lang_data.items()}
        return lang_data
    except (ImportError, FileNotFoundError, AttributeError):
        from locales import lang_en
        lang_data = {key.upper(): value for key, value in lang_en.lang_data.items()}
        return lang_data
locale_data = load_locale()

# デフォルトconfig.jsonの読み込み
with open(default_config_path, "r", encoding="utf-8") as f:
    default_config = json.load(f)

# IOフォルダのパスチェック関数
def path_check(input_folder, output_folder):
    if not input_folder or not os.path.exists(input_folder):
        gr.Warning(locale_data["LNG_INPUT_FOLDER_ALERT"])
        return False
    if not output_folder or not os.path.exists(output_folder):
        gr.Warning(locale_data["LNG_OUTPUT_FOLDER_ALERT"])
        return False
    return True

# 入力フォルダ直下に音声ファイルがあるかチェックする関数
def has_audio_files_in_root(input_folder):
    audio_extensions = ["*.wav", "*.flac", "*.ogg", "*.aiff", "*.mp3"]
    input_path = Path(input_folder)
    for ext in audio_extensions:
        if any(input_path.glob(ext)):
            return True
    return False

# 入力フォルダのサブフォルダに音声ファイルがあるかチェックする関数
def has_audio_files_in_subfolders(input_folder):
    audio_extensions = ["*.wav", "*.flac", "*.ogg", "*.aiff", "*.mp3"]
    input_path = Path(input_folder)
    for ext in audio_extensions:
        for f in input_path.glob(f"**/{ext}"):
            if f.parent != input_path:
                return True
    return False

# カンマ区切りの文字列をfloatのリストに変換するヘルパー関数
def str_to_float_list(s):
    if not isinstance(s, str):
        return s
    return [float(item.strip()) for item in s.split(',') if item.strip()]

# 設定ファイル生成用関数
def generate_config(
    # Training
    learning_rate_g, learning_rate_d, learning_rate_decay, adam_betas_1, adam_betas_2, adam_eps,
    batch_size, grad_weight_loudness, grad_weight_mel, grad_weight_ap, grad_weight_adv,
    grad_weight_fm, grad_balancer_ema_decay, use_amp, num_workers, n_steps, warmup_steps,
    evaluation_interval, save_interval,
    # Audio
    in_sample_rate, out_sample_rate, wav_length, segment_length, phone_noise_ratio, vq_topk,
    training_time_vq, floor_noise_level, record_metrics,
    # Augmentation
    aug_snr_candidates, aug_formant_shift_prob, aug_formant_shift_min, aug_formant_shift_max,
    aug_reverb_prob, aug_lpf_prob, aug_lpf_cutoff_candidates,
    # Data
    phone_extractor_file, pitch_estimator_file, in_ir_wav_dir, in_noise_wav_dir,
    in_test_wav_dir, pretrained_file,
    # Model
    pitch_bins, hidden_channels, san, compile_convnext, compile_d4c, compile_discriminator, profile,
    # Folders
    input_folder, output_folder
):
    # 設定ファイルの内容を辞書に格納
    config = {
        # training
        "learning_rate_g": float(learning_rate_g),
        "learning_rate_d": float(learning_rate_d),
        "learning_rate_decay": float(learning_rate_decay),
        "adam_betas": [float(adam_betas_1), float(adam_betas_2)],
        "adam_eps": float(adam_eps),
        "batch_size": int(batch_size),
        "grad_weight_loudness": float(grad_weight_loudness),
        "grad_weight_mel": float(grad_weight_mel),
        "grad_weight_ap": float(grad_weight_ap),
        "grad_weight_adv": float(grad_weight_adv),
        "grad_weight_fm": float(grad_weight_fm),
        "grad_balancer_ema_decay": float(grad_balancer_ema_decay),
        "use_amp": bool(use_amp),
        "num_workers": int(num_workers),
        "n_steps": int(n_steps),
        "warmup_steps": int(warmup_steps),
        "evaluation_interval": int(evaluation_interval),
        "save_interval": int(save_interval),
        # audio processing
        "in_sample_rate": int(in_sample_rate),
        "out_sample_rate": int(out_sample_rate),
        "wav_length": int(wav_length),
        "segment_length": int(segment_length),
        "phone_noise_ratio": float(phone_noise_ratio),
        "vq_topk": int(vq_topk),
        "training_time_vq": str(training_time_vq),
        "floor_noise_level": float(floor_noise_level),
        "record_metrics": bool(record_metrics),
        # augmentation
        "augmentation_snr_candidates": str_to_float_list(aug_snr_candidates),
        "augmentation_formant_shift_probability": float(aug_formant_shift_prob),
        "augmentation_formant_shift_semitone_min": float(aug_formant_shift_min),
        "augmentation_formant_shift_semitone_max": float(aug_formant_shift_max),
        "augmentation_reverb_probability": float(aug_reverb_prob),
        "augmentation_lpf_probability": float(aug_lpf_prob),
        "augmentation_lpf_cutoff_freq_candidates": str_to_float_list(aug_lpf_cutoff_candidates),
        # data paths
        "phone_extractor_file": phone_extractor_file,
        "pitch_estimator_file": pitch_estimator_file,
        "in_ir_wav_dir": in_ir_wav_dir,
        "in_noise_wav_dir": in_noise_wav_dir,
        "in_test_wav_dir": in_test_wav_dir,
        "pretrained_file": pretrained_file,
        # model architecture
        "pitch_bins": int(pitch_bins),
        "hidden_channels": int(hidden_channels),
        "san": bool(san),
        "compile_convnext": bool(compile_convnext),
        "compile_d4c": bool(compile_d4c),
        "compile_discriminator": bool(compile_discriminator),
        "profile": bool(profile)
    }
    # 設定ファイルをJSON形式で保存
    config_path = os.path.join(output_folder, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
        gr.Info(locale_data["LNG_CONFIG_SAVE_INFO"])
    # 環境変数を設定
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    return config_path

# トレーニングコマンドを実行する関数
def start_training_command(input_folder, output_folder, is_resume):
    global training_process
    config_path = os.path.join(output_folder, "config.json")
    command = [
        "python",
        "beatrice_trainer/__main__.py",
        "-d", input_folder,
        "-o", output_folder,
        "-c", config_path
    ]
    if is_resume:
        command.append("-r")
    # ノンブロッキングでプロセスを開始
    training_process = subprocess.Popen(command)

# トレーニングキューを処理するジェネレータ関数
def process_training_queue_generator():
    global training_process, is_terminated_by_user, training_tasks
    # 開始時にボタンを無効化
    yield gr.Button(interactive=False), gr.Button(interactive=False), gr.Button(interactive=True), gr.Markdown(display_queue())
    for task in training_tasks:
        if is_terminated_by_user:
            break
        if task['status'] == 'pending':
            task['status'] = 'in_progress'
            yield gr.Button(interactive=False), gr.Button(interactive=False), gr.Button(interactive=True), gr.Markdown(display_queue())
            input_folder = task['input_folder']
            output_folder = task['output_folder']
            is_resume = task['is_resume']
            config_params = task['config_params']
            gr.Info(locale_data["LNG_QUEUE_START_INFO"].format(output_folder=output_folder))
            # 設定ファイルを生成
            generate_config(*config_params, input_folder=input_folder, output_folder=output_folder)
            start_training_command(input_folder, output_folder, is_resume)
            while training_process.poll() is None:
                yield gr.Button(interactive=False), gr.Button(interactive=False), gr.Button(interactive=True), gr.Markdown(display_queue())
                time.sleep(1)
            if not is_terminated_by_user and training_process.poll() == 0:
                task['status'] = 'completed'
            elif is_terminated_by_user:
                task['status'] = 'stopped'
                break
            else:
                task['status'] = 'error'
                break
    # 全てのタスクが完了したか、停止した場合
    training_process = None
    if is_terminated_by_user:
        print(locale_data["LNG_USER_TERMINATED_MESSAGE"])
    else:
        print(locale_data["LNG_ALL_TASKS_COMPLETED_MESSAGE"])
    is_terminated_by_user = False
    
    # 終了後にボタンを有効化
    yield gr.Button(interactive=True), gr.Button(interactive=True), gr.Button(interactive=False), gr.Markdown(display_queue())

# 停止関数
def stop_training():
    global training_process, is_terminated_by_user, training_tasks
    is_terminated_by_user = True
    if training_process and training_process.poll() is None:
        training_process.kill()
        gr.Info(locale_data["LNG_STOP_SUCCESS_MESSAGE"])
    else:
        gr.Info(locale_data["LNG_STOP_NO_PROCESS_MESSAGE"])
    # キューを完全にクリアする
    training_tasks.clear()
    return gr.Button(interactive=True), gr.Button(interactive=False), gr.Button(interactive=False), gr.Markdown(display_queue())

# キューにタスクを追加する関数
def add_to_queue(input_folder, output_folder, checkpoint, *args):
    global training_tasks
    # 既存のパスチェック
    if not path_check(input_folder, output_folder):
        return gr.Markdown(display_queue()), gr.Button(interactive=True), gr.Button(interactive=False), gr.Button(interactive=False)
    # フォルダ直下に音声ファイルがあるかチェック
    if has_audio_files_in_root(input_folder):
        gr.Warning(locale_data["LNG_NO_AUDIO_FILES_IN_ROOT_ALERT"])
        return gr.Markdown(display_queue()), gr.Button(interactive=True), gr.Button(interactive=False), gr.Button(interactive=False)
    # サブフォルダに音声ファイルがあるかチェック
    if not has_audio_files_in_subfolders(input_folder):
        gr.Warning(locale_data["LNG_NO_AUDIO_FILES_IN_SUBFOLDERS_ALERT"])
        return gr.Markdown(display_queue()), gr.Button(interactive=True), gr.Button(interactive=False), gr.Button(interactive=False)
    # 追加学習かどうかを判断
    is_resume = False
    is_manual_checkpoint = False
    latest_checkpoint_path = os.path.join(output_folder, "checkpoint_latest.pt.gz")

    if checkpoint and os.path.isfile(os.path.join(output_folder, checkpoint)):
        # ユーザーが明示的にチェックポイントを指定した場合
        if os.path.isfile(latest_checkpoint_path):
            backup_path = os.path.join(output_folder, "checkpoint_latest.pt.gz.bk")
            shutil.copy2(latest_checkpoint_path, backup_path)
            gr.Info(locale_data["LNG_LATEST_CHECKPOINT_BACKUP"].format(backup_path=backup_path))
        shutil.copy2(os.path.join(output_folder, checkpoint), latest_checkpoint_path)
        is_resume = True
        is_manual_checkpoint = True
    elif os.path.isfile(latest_checkpoint_path):
        # チェックポイントの指定がない、または`checkpoint_latest.pt.gz`が指定された場合
        is_resume = True

    # 追加学習の場合、n_stepsが既存のステップ数より大きいかチェック
    if is_resume and not is_manual_checkpoint:
        config_path = os.path.join(output_folder, "config.json")
        if os.path.isfile(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    existing_config = json.load(f)
                existing_steps = existing_config.get("n_steps", 0)
                n_steps_val = args[15]
                if n_steps_val <= existing_steps:
                    gr.Warning(locale_data["LNG_RESUME_WARNING"].format(n_steps_val=n_steps_val, existing_steps=existing_steps))
            except Exception as e:
                gr.Warning(locale_data["LNG_CONFIG_LOAD_ERROR"].format(e=e))
        else:
            gr.Warning(locale_data["LNG_CONFIG_NOT_FOUND_WARNING"])
            is_resume = False
    task_data = {
        'input_folder': input_folder,
        'output_folder': output_folder,
        'is_resume': is_resume,
        'config_params': args,
        'status': 'pending'
    }
    training_tasks.append(task_data)
    if is_resume:
        gr.Info(locale_data["LNG_QUEUE_TASK_RESUMED"].format(output_folder=output_folder))
    else:
        gr.Info(locale_data["LNG_QUEUE_TASK_NEW"].format(output_folder=output_folder))
    return gr.Markdown(display_queue()), gr.Button(interactive=True), gr.Button(interactive=True), gr.Button(interactive=True)

# キューの表示を更新する関数
def display_queue():
    global training_tasks
    if not training_tasks:
        return locale_data["LNG_TASK_MONITOR_EMPTY"]
    display_text = ""
    for i, task in enumerate(training_tasks):
        status_text = ""
        if task['status'] == 'pending':
            status_text = locale_data["LNG_QUEUE_STATUS_PENDING"]
        elif task['status'] == 'in_progress':
            status_text = locale_data["LNG_QUEUE_STATUS_IN_PROGRESS"]
        elif task['status'] == 'completed':
            status_text = locale_data["LNG_QUEUE_STATUS_COMPLETED"]
        elif task['status'] == 'error':
            status_text = locale_data["LNG_QUEUE_STATUS_ERROR"]
        elif task['status'] == 'stopped':
            status_text = locale_data["LNG_QUEUE_STATUS_STOPPED"]
        display_text += f"{i+1:02d} | {status_text} | `{task['output_folder']}` | n_steps: `{task['config_params'][15]}` | batch_size: `{task['config_params'][6]}`  \n"
    return display_text

# TensorBoardを起動する関数
def start_tensorboard(output_folder):
    if output_folder:
        subprocess.run(["taskkill", "/F", "/IM", "tensorboard.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        command = ["tensorboard", "--logdir", output_folder]
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        webbrowser.open("http://localhost:6006")

def stop_slicing_process():
    global is_slicing_terminated_by_user
    is_slicing_terminated_by_user = True
    gr.Info(locale_data["LNG_STATUS_STOPPED"])
    return gr.Button(interactive=True), gr.Button(interactive=False), locale_data['LNG_STATUS_STOPPED']

# 音声ファイルの前処理関数
def remove_silence(waveform, sample_rate, silence_threshold_dbfs, min_silence_duration_ms):
    frame_size = int(0.02 * sample_rate)
    hop_size = frame_size // 2
    num_frames = max(1, (waveform.size(1) - frame_size) // hop_size + 1)
    energy = []
    for i in range(num_frames):
        start = i * hop_size
        frame = waveform[:, start:start+frame_size]
        rms = torch.sqrt(torch.mean(frame**2))
        rms_db = 20 * torch.log10(rms + 1e-9)
        energy.append(rms_db.item())
    energy = torch.tensor(energy)
    is_silent = energy < silence_threshold_dbfs
    min_silence_frames = int((min_silence_duration_ms / 1000) * sample_rate / hop_size)
    keep_samples = torch.ones(waveform.size(1), dtype=torch.bool)
    silent_run = 0
    run_start = 0
    for i, silent in enumerate(is_silent):
        if silent:
            if silent_run == 0:
                run_start = i
            silent_run += 1
        else:
            if silent_run >= min_silence_frames:
                start_sample = run_start * hop_size
                end_sample = (i * hop_size) + frame_size
                keep_samples[start_sample:end_sample] = False
            silent_run = 0
    if silent_run >= min_silence_frames:
        start_sample = run_start * hop_size
        end_sample = waveform.size(1)
        keep_samples[start_sample:end_sample] = False
    return waveform[:, keep_samples]
    
# 音声ファイルの分割処理関数
def process_single_folder(input_folder, output_dir, duration, sample_rate, output_format, enable_silence_removal, silence_threshold_dbfs, min_silence_duration_ms):
    global is_slicing_terminated_by_user
    audio_extensions = ["*.wav", "*.flac", "*.ogg", "*.aiff", "*.mp3"]
    audio_files = []
    input_path = Path(input_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in audio_extensions:
        audio_files.extend(list(input_path.glob(ext)))
    
    if not audio_files:
        yield 0, 0, []
    
    file_counter_in = 1
    processed_count_local = 0
    skipped_count_local = 0
    failed_files_local = []
    
    for audio_path in audio_files:
        if is_slicing_terminated_by_user:
            break
        
        # ファイルごとにステータスをyieldで返すように修正
        yield locale_data['LNG_STATUS_SLICE'], audio_path.name
        
        try:
            waveform, original_sr = torchaudio.load(audio_path)
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if original_sr != int(sample_rate):
                resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=int(sample_rate))
                waveform = resampler(waveform)
            if enable_silence_removal:
                processed_waveform = remove_silence(
                    waveform, int(sample_rate), silence_threshold_dbfs, min_silence_duration_ms
                )
            else:
                processed_waveform = waveform
            
            total_samples = processed_waveform.size(1)
            segment_length_samples = int(duration * int(sample_rate))

            if total_samples < segment_length_samples:
                skipped_count_local += 1
                continue
                
            start_sample = 0
            file_counter = 1
            while start_sample + segment_length_samples <= total_samples:
                segment_data = processed_waveform[:, start_sample : start_sample + segment_length_samples]
                new_file_prefix = f"{file_counter_in:04d}"
                file_name = f"{new_file_prefix}_{file_counter:04d}.{output_format}"
                output_file_path = output_dir / file_name
                torchaudio.save(output_file_path, segment_data, int(sample_rate), format=output_format)
                start_sample += segment_length_samples
                file_counter += 1
            file_counter_in += 1
            processed_count_local += 1

        except Exception as e:
            failed_files_local.append(f"{audio_path.name}: {e}")
            
    yield processed_count_local, skipped_count_local, failed_files_local

# メインの分割処理関数
def split_audio_files(input_folder, output_folder, enable_whole_dataset_slice, duration, sample_rate, output_format, enable_silence_removal, silence_threshold_dbfs, min_silence_duration_ms):
    global is_slicing_terminated_by_user
    is_slicing_terminated_by_user = False
    yield gr.Button(interactive=False), gr.Button(interactive=True), locale_data["LNG_STATUS_WAITING"]

    if not input_folder:
        gr.Warning(locale_data["LNG_ERROR_NO_INPUT_FOLDER"])
        yield gr.Button(interactive=True), gr.Button(interactive=False), locale_data["LNG_STATUS_WAITING"]
        return
    
    input_path = Path(input_folder)
    if not input_path.exists():
        gr.Warning(locale_data["LNG_INPUT_FOLDER_ALERT"])
        yield gr.Button(interactive=True), gr.Button(interactive=False), locale_data["LNG_STATUS_WAITING"]
        return

    total_processed_count = 0
    total_skipped_count = 0
    all_failed_files = []
    
    if enable_whole_dataset_slice:
        if not has_audio_files_in_subfolders(input_folder):
            gr.Warning(locale_data["LNG_NO_AUDIO_FILES_IN_SUBFOLDERS_ALERT"])
            yield gr.Button(interactive=True), gr.Button(interactive=False), locale_data["LNG_STATUS_WAITING"]
            return
            
        if not output_folder:
            output_folder = Path(input_folder).parent / f"{Path(input_folder).name}_slice"
            output_folder.mkdir(parents=True, exist_ok=True)
        else:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
            
        subfolders = [f for f in input_path.iterdir() if f.is_dir()]
        total_folders = len(subfolders)
        for i, subfolder in enumerate(subfolders):
            if is_slicing_terminated_by_user:
                break
            # フォルダごとのステータスを更新
            yield gr.Button(interactive=False), gr.Button(interactive=True), f"{locale_data['LNG_STATUS_SLICE']} {subfolder.name} ({i+1}/{total_folders})"
            relative_path = subfolder.relative_to(input_path)
            output_subfolder = output_folder / relative_path
            
            # process_single_folderのジェネレータをループしてステータスを中継
            for status_or_result in process_single_folder(subfolder, output_subfolder, duration, sample_rate, output_format, enable_silence_removal, silence_threshold_dbfs, min_silence_duration_ms):
                if isinstance(status_or_result, tuple) and len(status_or_result) == 2:
                    status, filename = status_or_result
                    yield gr.Button(interactive=False), gr.Button(interactive=True), f"{status} {filename}"
                else:
                    processed, skipped, failed = status_or_result
                    total_processed_count += processed
                    total_skipped_count += skipped
                    all_failed_files.extend(failed)
    else: # 単一フォルダモード
        if not output_folder:
            base_output_dir_name = f"slice_{output_format}"
            output_folder = input_path / base_output_dir_name
            counter = 1
            while output_folder.exists():
                counter += 1
                output_folder = input_path / f"{base_output_dir_name}_{counter:03d}"
        else:
            output_folder = Path(output_folder)
        
        # 音声ファイルが見つかるか再チェック
        audio_extensions = ["*.wav", "*.flac", "*.ogg", "*.aiff", "*.mp3"]
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(list(input_path.glob(f"**/{ext}")))
        if not audio_files:
            gr.Warning(locale_data["LNG_WARNING_NO_AUDIO_FILES"])
            yield gr.Button(interactive=True), gr.Button(interactive=False), locale_data["LNG_STATUS_WAITING"]
            return
        
        # process_single_folderのジェネレータをループしてステータスを中継
        for status_or_result in process_single_folder(input_folder, output_folder, duration, sample_rate, output_format, enable_silence_removal, silence_threshold_dbfs, min_silence_duration_ms):
            if isinstance(status_or_result, tuple) and len(status_or_result) == 2:
                status, filename = status_or_result
                yield gr.Button(interactive=False), gr.Button(interactive=True), f"{status} {filename}"
            else:
                processed, skipped, failed = status_or_result
                total_processed_count = processed
                total_skipped_count = skipped
                all_failed_files.extend(failed)

    if is_slicing_terminated_by_user:
        yield gr.Button(interactive=True), gr.Button(interactive=False), locale_data['LNG_STATUS_STOPPED']
    elif all_failed_files:
        failed_message = "<br>".join(all_failed_files)
        yield gr.Button(interactive=True), gr.Button(interactive=False), locale_data['LNG_COMPLETE_WITH_FAILURES'].format(processed_count=total_processed_count, len=len(all_failed_files), failed_message=failed_message)
    else:
        yield gr.Button(interactive=True), gr.Button(interactive=False), locale_data['LNG_COMPLETE_SUCCESS'].format(processed_count=total_processed_count, skipped_count=total_skipped_count, output_dir=output_folder)


# UI構築
with gr.Blocks() as demo:
    gr.HTML(f"<h1>{locale_data['LNG_TITLE']}</h1><p style='font-size: 1.0em;'>Ver: {VERSION}</p>")
    with gr.Tabs() as tabs:
        # --- トレーニングタブ ---
        with gr.Tab(locale_data["LNG_TAB_TRAIN"]):
            gr.Markdown(locale_data["LNG_TRAIN_DESC"])
            # --- 基本トレーニング設定 ---
            with gr.Row():
                with gr.Column():
                    input_folder = gr.Textbox(
                        label=locale_data["LNG_INPUT_FOLDER"],
                        placeholder=locale_data["LNG_INPUT_FOLDER_PLACE"],
                        info=locale_data["LNG_INPUT_FOLDER_INFO"]
                    )
                    output_folder = gr.Textbox(
                        label=locale_data["LNG_OUTPUT_FOLDER"],
                        placeholder=locale_data["LNG_OUTPUT_FOLDER_PLACE"]
                    )
                    checkpoint = gr.Textbox(
                        label=locale_data["LNG_CHECKPOINT"],
                        placeholder=locale_data["LNG_CHECKPOINT_PLACE"]
                    )
            # --- Main Training Parameters---
            with gr.Accordion(locale_data["LNG_BASIC_TRAINING"], open=True):
                with gr.Row():
                    n_steps = gr.Number(
                        value=default_config["n_steps"],
                        label="n_steps",
                        info=locale_data["LNG_N_STEPS_INFO"]
                    )
                    batch_size = gr.Slider(
                        minimum=1,maximum=64,step=1,
                        value=default_config["batch_size"],
                        label="Batch Size",
                        info=locale_data["LNG_BATCH_SIZE_INFO"]
                    )
                    num_workers = gr.Slider(
                        minimum=0,maximum=64,step=1,
                        value=default_config["num_workers"],
                        label="Num Workers",
                        info=locale_data["LNG_NUM_WORKERS_INFO"]
                    )
                with gr.Row():
                    save_interval = gr.Number(
                        value=default_config["save_interval"],
                        label="Save Interval",
                        info=locale_data["LNG_SAVE_INTERVAL_INFO"]
                    )
                    evaluation_interval = gr.Number(
                        value=default_config["evaluation_interval"],
                        label="Evaluation Interval",
                        info=locale_data["LNG_EVALUATION_INTERVAL_INFO"]
                    )
            # --- 詳細設定 ---
            with gr.Accordion(locale_data["LNG_ADVANCED_OPTIONS"], open=False):
                # --- Learning Rate / Optimizer ---
                with gr.Accordion(locale_data["LNG_LEARNING_RATE_OPTIMIZER"], open=False):
                    with gr.Row():
                        learning_rate_g = gr.Number(label="Learning Rate G", value=default_config["learning_rate_g"])
                        learning_rate_d = gr.Number(label="Learning Rate D", value=default_config["learning_rate_d"])
                        learning_rate_decay = gr.Number(label="Learning Rate Decay", value=default_config["learning_rate_decay"])
                        warmup_steps = gr.Number(label="Warmup Steps", value=default_config["warmup_steps"])
                    with gr.Row():
                        adam_betas_1 = gr.Number(label="Adam Betas 1", value=default_config["adam_betas"][0])
                        adam_betas_2 = gr.Number(label="Adam Betas 2", value=default_config["adam_betas"][1])
                        adam_eps = gr.Number(label="Adam Eps", value=default_config["adam_eps"])
                # --- Loss Weights ---
                with gr.Accordion(locale_data["LNG_LOSS_WEIGHTS"], open=False):
                    with gr.Row():
                        grad_weight_loudness = gr.Number(label="Grad Weight Loudness", value=default_config["grad_weight_loudness"], step=0.1, precision=2)
                        grad_weight_mel = gr.Number(label="Grad Weight Mel", value=default_config["grad_weight_mel"], step=0.1, precision=2)
                        grad_weight_ap = gr.Number(label="Grad Weight AP", value=default_config["grad_weight_ap"], step=0.1, precision=2)
                    with gr.Row():
                        grad_weight_adv = gr.Number(label="Grad Weight Adv", value=default_config["grad_weight_adv"], step=0.1, precision=2)
                        grad_weight_fm = gr.Number(label="Grad Weight FM", value=default_config["grad_weight_fm"], step=0.1, precision=2)
                        grad_balancer_ema_decay = gr.Number(label="Grad Balancer EMA Decay", value=default_config["grad_balancer_ema_decay"])
                # --- Augmentation options ---
                with gr.Accordion(locale_data["LNG_AUGMENTATION_OPTIONS"], open=False):
                    with gr.Row():
                        aug_formant_shift_prob = gr.Slider(minimum=0.0, maximum=1.0, label="Formant Shift Probability", value=default_config["augmentation_formant_shift_probability"])
                        aug_formant_shift_min = gr.Number(label="Formant Shift Semitone Min", value=default_config["augmentation_formant_shift_semitone_min"])
                        aug_formant_shift_max = gr.Number(label="Formant Shift Semitone Max", value=default_config["augmentation_formant_shift_semitone_max"])
                    with gr.Row():
                        aug_reverb_prob = gr.Slider(minimum=0.0, maximum=1.0, label="Reverb Probability", value=default_config["augmentation_reverb_probability"])
                        aug_lpf_prob = gr.Slider(minimum=0.0, maximum=1.0, label="LPF Probability", value=default_config["augmentation_lpf_probability"])
                    with gr.Row():
                        aug_snr_candidates = gr.Textbox(label="SNR Candidates (comma-separated)", value=", ".join(map(str, default_config["augmentation_snr_candidates"])))
                        aug_lpf_cutoff_candidates = gr.Textbox(label="LPF Cutoff Freq Candidates (comma-separated)", value=", ".join(map(str, default_config["augmentation_lpf_cutoff_freq_candidates"])))
                # --- Audio / Model ---
                with gr.Accordion(locale_data["LNG_AUDIO_MODEL"], open=False):
                    with gr.Row():
                        in_sample_rate = gr.Number(label="In Sample Rate", value=default_config["in_sample_rate"], interactive=False, info=locale_data["LNG_IN_SAMPLE_RATE_INFO"])
                        out_sample_rate = gr.Number(label="Out Sample Rate", value=default_config["out_sample_rate"], interactive=False, info=locale_data["LNG_OUT_SAMPLE_RATE_INFO"])
                        wav_length = gr.Number(label="Wav Length", value=default_config["wav_length"])
                        segment_length = gr.Number(label="Segment Length", value=default_config["segment_length"])
                    with gr.Row():
                        phone_noise_ratio = gr.Slider(minimum=0.0, maximum=1.0, label="Phone Noise Ratio", value=default_config["phone_noise_ratio"])
                        floor_noise_level = gr.Number(label="Floor Noise Level", value=default_config["floor_noise_level"])
                        vq_topk = gr.Number(label="VQ Top-K", value=default_config["vq_topk"], step=1)
                        training_time_vq = gr.Dropdown(label="Training Time VQ", choices=["none", "self", "random"], value=default_config["training_time_vq"])
                    with gr.Row():
                        pitch_bins = gr.Number(label="Pitch Bins", value=default_config["pitch_bins"], interactive=False)
                        hidden_channels = gr.Number(label="Hidden Channels", value=default_config["hidden_channels"])
                # --- File Paths ---
                with gr.Accordion(locale_data["LNG_FILE_PATHS"], open=False):
                    with gr.Column():
                        in_ir_wav_dir = gr.Textbox(label="In IR Wav Dir", value=default_config["in_ir_wav_dir"])
                        in_noise_wav_dir = gr.Textbox(label="In Noise Wav Dir", value=default_config["in_noise_wav_dir"])
                        in_test_wav_dir = gr.Textbox(label="In Test Wav Dir", value=default_config["in_test_wav_dir"])
                        pretrained_file = gr.Textbox(label="Pretrained File", value=default_config["pretrained_file"])
                        phone_extractor_file = gr.Textbox(label="Phone Extractor File", value=default_config["phone_extractor_file"])
                        pitch_estimator_file = gr.Textbox(label="Pitch Estimator File", value=default_config["pitch_estimator_file"])
                # --- Performance / Debug ---
                with gr.Accordion(locale_data["LNG_PERFORMANCE_DEBUG"], open=False):
                    with gr.Row():
                        with gr.Column():
                            use_amp = gr.Checkbox(label="Use AMP", value=default_config["use_amp"])
                        with gr.Column():
                            san = gr.Checkbox(label="SAN (Discriminator)", value=default_config["san"])
                        with gr.Column():
                            record_metrics = gr.Checkbox(label="Record Metrics to TensorBoard", value=default_config["record_metrics"])
                        with gr.Column():
                            profile = gr.Checkbox(label="Profile", value=default_config["profile"])
                    with gr.Row():
                        compile_convnext = gr.Checkbox(label="Compile ConvNext", value=default_config["compile_convnext"])
                        compile_d4c = gr.Checkbox(label="Compile D4c", value=default_config["compile_d4c"])
                        compile_discriminator = gr.Checkbox(label="Compile Discriminator", value=default_config["compile_discriminator"])
            # --- ボタン類 ---
            with gr.Row():
                add_to_queue_button = gr.Button(value=locale_data["LNG_ADD_TASK_BUTTON"], variant="primary")
                start_button = gr.Button(value=locale_data["LNG_START_TRAINING_BUTTON"], variant="primary", interactive=False)
                stop_button = gr.Button(value=locale_data["LNG_STOP_TRAINING_BUTTON"], variant="stop", interactive=False)
                tensorboard_button = gr.Button(value=locale_data["LNG_TENSORBOARD_BUTTON"])
                
            # キューの状態表示
            queue_status_box = gr.Markdown(locale_data["LNG_TASK_MONITOR_EMPTY"])
            
            # --- Event Handlers ---
            all_inputs = [
                # Training
                learning_rate_g, learning_rate_d, learning_rate_decay, adam_betas_1, adam_betas_2, adam_eps,
                batch_size, grad_weight_loudness, grad_weight_mel, grad_weight_ap, grad_weight_adv,
                grad_weight_fm, grad_balancer_ema_decay, use_amp, num_workers, n_steps, warmup_steps,
                evaluation_interval, save_interval,
                # Audio
                in_sample_rate, out_sample_rate, wav_length, segment_length, phone_noise_ratio, vq_topk,
                training_time_vq, floor_noise_level, record_metrics,
                # Augmentation
                aug_snr_candidates, aug_formant_shift_prob, aug_formant_shift_min, aug_formant_shift_max,
                aug_reverb_prob, aug_lpf_prob, aug_lpf_cutoff_candidates,
                # Data
                phone_extractor_file, pitch_estimator_file, in_ir_wav_dir, in_noise_wav_dir,
                in_test_wav_dir, pretrained_file,
                # Model
                pitch_bins, hidden_channels, san, compile_convnext, compile_d4c, compile_discriminator, profile
            ]
            
            add_to_queue_button.click(add_to_queue, inputs=[input_folder, output_folder, checkpoint] + all_inputs, outputs=[queue_status_box, add_to_queue_button, start_button, stop_button])
            start_button.click(process_training_queue_generator, inputs=None, outputs=[start_button, add_to_queue_button, stop_button, queue_status_box])
            stop_button.click(stop_training, inputs=None, outputs=[add_to_queue_button, start_button, stop_button, queue_status_box])
            tensorboard_button.click(
                lambda output_folder_val: gr.Warning(locale_data["LNG_TENSORBOARD_ALERT"]) if not output_folder_val else start_tensorboard(output_folder_val),
                inputs=[output_folder],
                outputs=None,
            )
            
        # --- 「データセットの前処理」タブ ---
        with gr.Tab(locale_data["LNG_TAB_DATASET_PROCESSING"]):
            gr.Markdown(locale_data["LNG_DATASET_PROCESSING_DESC"])
            with gr.Column():
                input_dir_prep = gr.Textbox(
                    label=locale_data["LNG_INPUT_DIR_PREP_LABEL"],
                    placeholder=locale_data["LNG_INPUT_DIR_PREP_PLACE"]
                )
                output_dir_prep = gr.Textbox(
                    label=locale_data["LNG_OUTPUT_DIR_PREP_LABEL"],
                    placeholder=locale_data["LNG_OUTPUT_DIR_PREP_PLACE"]
                )
                enable_whole_dataset_slice = gr.Checkbox(
                    info=locale_data["LNG_DATASET_SLICE_INFO"],
                    label=locale_data["LNG_DATASET_SLICE"],
                    value=False
                )
                with gr.Row():
                    segment_duration_sec = gr.Slider(
                        minimum=1,maximum=30,step=1,value=4,
                        label=locale_data["LNG_SEGMENT_DURATION_LABEL"],
                        info=locale_data["LNG_SEGMENT_DURATION_INFO"]
                    )
                    output_samplerate = gr.Dropdown(
                        label=locale_data["LNG_OUTPUT_SAMPLERATE_LABEL"],
                        choices=["8000", "11025", "16000", "22050", "32000", "44100", "48000"],
                        value="16000",
                        info=locale_data["LNG_OUTPUT_SAMPLERATE_INFO"],
                    )
                    output_format = gr.Dropdown(
                        label=locale_data["LNG_OUTPUT_FORMAT_LABEL"],
                        choices=["wav", "flac"],
                        value="wav",
                        info=locale_data["LNG_OUTPUT_FORMAT_INFO"]
                    )
                with gr.Row():
                    enable_silence_removal = gr.Checkbox(
                        label=locale_data["LNG_SILENCE_REMOVAL_OPTIONS"],
                        info=locale_data["LNG_SILENCE_REMOVAL_OPTIONS_INFO"],
                        value=True
                    )
                    silence_threshold_dbfs = gr.Slider(
                        minimum=-60,maximum=0,step=1,value=-40,
                        label=locale_data["LNG_SILENCE_THRESHOLD_LABEL"],
                        info=locale_data["LNG_SILENCE_THRESHOLD_INFO"]
                    )
                    min_silence_duration_ms = gr.Slider(
                        minimum=100,maximum=5000,step=100,value=500,
                        label=locale_data["LNG_MIN_SILENCE_DURATION_LABEL"],
                        info=locale_data["LNG_MIN_SILENCE_DURATION_INFO"]
                    )             
                with gr.Row():
                    split_button = gr.Button(locale_data["LNG_SPLIT_BUTTON"], variant="primary")
                    stop_slice_button = gr.Button(locale_data["LNG_STOP_SLICE_BUTTON"], variant="stop", interactive=False)
            
            status_markdown = gr.Markdown(locale_data["LNG_STATUS_WAITING"])
            
            split_button.click(
                fn=split_audio_files,
                inputs=[input_dir_prep, output_dir_prep, enable_whole_dataset_slice, segment_duration_sec, output_samplerate, output_format, enable_silence_removal, silence_threshold_dbfs, min_silence_duration_ms],
                outputs=[split_button, stop_slice_button, status_markdown],
            )
            stop_slice_button.click(
                fn=stop_slicing_process,
                inputs=None,
                outputs=[split_button, stop_slice_button, status_markdown],
            )
            
demo.launch(inbrowser=True)