import gradio as gr
import json
import os
import subprocess
import webbrowser
import locale
import importlib
import shutil

# バージョン情報
VERSION = "2025.09.12"

# カレントディレクトリの取得
current_dir = os.getcwd()
default_config_path = os.path.join(current_dir, "assets", "default_config.json")

# 言語設定の読み込み
def load_locale():
    try:
        lang_code, _ = locale.getdefaultlocale()
        # 日本語ならlang_ja.py、それ以外ならlang_en.pyを読み込む
        lang_file = "lang_ja" if lang_code == "ja_JP" else "lang_en"
        lang_module = importlib.import_module(f"locales.{lang_file}")
        return lang_module.lang_data
    except (ImportError, FileNotFoundError):
        # デフォルトとして英語をフォールバック
        from locales import lang_en
        return lang_en.lang_data

locale_data = load_locale()

# デフォルトconfig.jsonの読み込み
with open(default_config_path, "r", encoding="utf-8") as f:
    default_config = json.load(f)

# IOフォルダのパスチェック関数
def path_check(input_folder, output_folder):
    if not input_folder:
        gr.Warning(locale_data["input_folder_alert1"])
        return False
    if not os.path.exists(input_folder):
        gr.Warning(locale_data["input_folder_alert2"])
        return False
    if not output_folder:
        gr.Warning(locale_data["output_folder_alert1"])
        return False
    # 出力フォルダが存在しない場合は作成する
    os.makedirs(output_folder, exist_ok=True)
    
    return True

# グローバル変数としてオプションを定義
add_option = ""  # デフォルトは空の文字列

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
        gr.Info(locale_data["config_save_info"])
    
    # 環境変数を設定
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# トレーニングコマンドを実行する関数
def run_training(input_folder, output_folder, checkpoint):
    config_path = os.path.join(output_folder, "config.json")
    latest_checkpoint_path = os.path.join(output_folder, "checkpoint_latest.pt.gz")

    add_option = ""
    # ユーザー入力またはデフォルトのチェックポイントの存在をチェック
    if checkpoint.lower() == "checkpoint_latest.pt.gz" and os.path.isfile(latest_checkpoint_path):
        add_option = "-r"
    elif checkpoint and os.path.isfile(os.path.join(output_folder, checkpoint)):
        gr.Info(locale_data["rename_info"].format(src=os.path.join(output_folder, checkpoint), dest=latest_checkpoint_path))
        shutil.copy2(os.path.join(output_folder, checkpoint), latest_checkpoint_path)
        add_option = "-r"
    elif not checkpoint and os.path.isfile(latest_checkpoint_path):
        add_option = "-r"
    
    if add_option == "-r":
        gr.Info(locale_data["addtrain_start"])
    else:
        gr.Info(locale_data["train_start"])
    
    command = [
        "python",
        "beatrice_trainer/__main__.py",
        "-d", input_folder,
        "-o", output_folder,
        "-c", config_path
    ]
    
    if add_option == "-r":
        command.append("-r")
        
    subprocess.run(command)

# 入力フィールドをリセットする関数
def reset_inputs():
    # default_configからリストを文字列に変換
    aug_snr_str = ", ".join(map(str, default_config["augmentation_snr_candidates"]))
    aug_lpf_str = ", ".join(map(str, default_config["augmentation_lpf_cutoff_freq_candidates"]))

    return [
        "",  # input_folder
        "",  # output_folder
        "",  # checkpoint
        # Training
        default_config["learning_rate_g"],
        default_config["learning_rate_d"],
        default_config["learning_rate_decay"],
        default_config["adam_betas"][0],
        default_config["adam_betas"][1],
        default_config["adam_eps"],
        default_config["batch_size"],
        default_config["grad_weight_loudness"],
        default_config["grad_weight_mel"],
        default_config["grad_weight_ap"],
        default_config["grad_weight_adv"],
        default_config["grad_weight_fm"],
        default_config["grad_balancer_ema_decay"],
        default_config["use_amp"],
        default_config["num_workers"],
        default_config["n_steps"],
        default_config["warmup_steps"],
        default_config["evaluation_interval"],
        default_config["save_interval"],
        # Audio
        default_config["in_sample_rate"],
        default_config["out_sample_rate"],
        default_config["wav_length"],
        default_config["segment_length"],
        default_config["phone_noise_ratio"],
        default_config["vq_topk"],
        default_config["training_time_vq"],
        default_config["floor_noise_level"],
        default_config["record_metrics"],
        # Augmentation
        aug_snr_str,
        default_config["augmentation_formant_shift_probability"],
        default_config["augmentation_formant_shift_semitone_min"],
        default_config["augmentation_formant_shift_semitone_max"],
        default_config["augmentation_reverb_probability"],
        default_config["augmentation_lpf_probability"],
        aug_lpf_str,
        # Data
        default_config["phone_extractor_file"],
        default_config["pitch_estimator_file"],
        default_config["in_ir_wav_dir"],
        default_config["in_noise_wav_dir"],
        in_test_wav_dir, default_config["pretrained_file"],
        # Model
        default_config["pitch_bins"],
        default_config["hidden_channels"],
        default_config["san"],
        default_config["compile_convnext"],
        default_config["compile_d4c"],
        default_config["compile_discriminator"],
        default_config["profile"]
    ]

# TensorBoardを起動する関数
def start_tensorboard(output_folder):
    if output_folder:
        subprocess.run(["taskkill", "/F", "/IM", "tensorboard.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        command = ["tensorboard", "--logdir", output_folder]
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        webbrowser.open("http://localhost:6006")

# UI構築
with gr.Blocks() as demo:
    gr.HTML(f"<h1>{locale_data['title']}</h1><p style='font-size: 1.0em;'>Ver: {VERSION}</p>")

    # --- Basic Settings ---
    with gr.Row():
        with gr.Column():
            input_folder = gr.Textbox(label=locale_data["input_folder"], placeholder=locale_data["input_folder_place"], info=locale_data["input_folder_info"])
            output_folder = gr.Textbox(label=locale_data["output_folder"], placeholder=locale_data["output_folder_place"], info=locale_data["output_folder_info"])
            checkpoint = gr.Textbox(label=locale_data["checkpoint"], placeholder=locale_data["checkpoint_place"], info=locale_data["checkpoint_info"])
    
    # --- Main Training Parameters (まとめてAccordion化) ---
    with gr.Accordion(locale_data["basic_training"], open=True):
        with gr.Row():
            n_steps = gr.Number(label="n_steps", minimum=1, step=1, value=default_config["n_steps"], interactive=True, info=locale_data["n_steps_info"])
            batch_size = gr.Number(label="Batch Size", minimum=1, step=1, value=default_config["batch_size"], interactive=True, info=locale_data["batch_size_info"])
            num_workers = gr.Number(label="Num Workers", minimum=0, step=1, value=default_config["num_workers"], interactive=True, info=locale_data["num_workers_info"])
        with gr.Row():
            save_interval = gr.Number(label="Save Interval", minimum=1, step=1, value=default_config["save_interval"], interactive=True, info=locale_data["save_interval_info"])
            evaluation_interval = gr.Number(label="Evaluation Interval", minimum=1, step=1, value=default_config["evaluation_interval"], interactive=True, info=locale_data["evaluation_interval_info"])
        
    # --- Advanced options ---
    with gr.Accordion(locale_data["advanced_options"], open=False):
        
        # --- Learning Rate / Optimizer ---
        with gr.Accordion(locale_data["learning_rate_optimizer"], open=False):
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
        with gr.Accordion(locale_data["loss_weights"], open=False):
            with gr.Row():
                grad_weight_loudness = gr.Number(label="Grad Weight Loudness", value=default_config["grad_weight_loudness"], step=0.1, precision=2)
                grad_weight_mel = gr.Number(label="Grad Weight Mel", value=default_config["grad_weight_mel"], step=0.1, precision=2)
                grad_weight_ap = gr.Number(label="Grad Weight AP", value=default_config["grad_weight_ap"], step=0.1, precision=2)
            with gr.Row():
                grad_weight_adv = gr.Number(label="Grad Weight Adv", value=default_config["grad_weight_adv"], step=0.1, precision=2)
                grad_weight_fm = gr.Number(label="Grad Weight FM", value=default_config["grad_weight_fm"], step=0.1, precision=2)
                grad_balancer_ema_decay = gr.Number(label="Grad Balancer EMA Decay", value=default_config["grad_balancer_ema_decay"])

        # --- Augmentation options ---
        with gr.Accordion(locale_data["augmentation_options"], open=False):
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
        with gr.Accordion(locale_data["audio_model"], open=False):
            with gr.Row():
                in_sample_rate = gr.Number(label="In Sample Rate", value=default_config["in_sample_rate"], interactive=False, info=locale_data["in_sample_rate_info"])
                out_sample_rate = gr.Number(label="Out Sample Rate", value=default_config["out_sample_rate"], interactive=False, info=locale_data["out_sample_rate_info"])
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
        with gr.Accordion(locale_data["file_paths"], open=False):
            with gr.Column():
                in_ir_wav_dir = gr.Textbox(label="In IR Wav Dir", value=default_config["in_ir_wav_dir"])
                in_noise_wav_dir = gr.Textbox(label="In Noise Wav Dir", value=default_config["in_noise_wav_dir"])
                in_test_wav_dir = gr.Textbox(label="In Test Wav Dir", value=default_config["in_test_wav_dir"])
                pretrained_file = gr.Textbox(label="Pretrained File", value=default_config["pretrained_file"])
                phone_extractor_file = gr.Textbox(label="Phone Extractor File", value=default_config["phone_extractor_file"])
                pitch_estimator_file = gr.Textbox(label="Pitch Estimator File", value=default_config["pitch_estimator_file"])

        # --- Performance / Debug ---
        with gr.Accordion(locale_data["performance_debug"], open=False):
            with gr.Row():
                with gr.Column():
                    use_amp = gr.Checkbox(label="Use AMP", value=default_config["use_amp"])
                    gr.Markdown(locale_data["use_amp_info"])
                with gr.Column():
                    san = gr.Checkbox(label="SAN (Discriminator)", value=default_config["san"])
                    gr.Markdown(locale_data["san_info"])
                with gr.Column():
                    record_metrics = gr.Checkbox(label="Record Metrics to TensorBoard", value=default_config["record_metrics"])
                    gr.Markdown(locale_data["record_metrics_info"])
                with gr.Column():
                    profile = gr.Checkbox(label="Profile", value=default_config["profile"])
                    gr.Markdown(locale_data["profile_info"])
            with gr.Row():
                compile_convnext = gr.Checkbox(label="Compile ConvNext", value=default_config["compile_convnext"])
                compile_d4c = gr.Checkbox(label="Compile D4c", value=default_config["compile_d4c"])
                compile_discriminator = gr.Checkbox(label="Compile Discriminator", value=default_config["compile_discriminator"])

    # --- Buttons ---
    with gr.Row():
        reset_button = gr.Button(value=locale_data["reset"])
        tensorboard_button = gr.Button(value=locale_data["tensorboard"])
    with gr.Column():
        train_button = gr.Button(value=locale_data["train"], variant="primary")

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

    train_button.click(
        lambda input_folder_val, output_folder_val, checkpoint_val, *args: (
            (generate_config(*args, input_folder_val, output_folder_val) or run_training(input_folder_val, output_folder_val, checkpoint_val))
            if path_check(input_folder_val, output_folder_val) else None
        ),
        inputs=[input_folder, output_folder, checkpoint] + all_inputs,
        outputs=None,
    )
    
    all_outputs = [
        input_folder, output_folder, checkpoint,
        learning_rate_g, learning_rate_d, learning_rate_decay, adam_betas_1, adam_betas_2, adam_eps,
        batch_size, grad_weight_loudness, grad_weight_mel, grad_weight_ap, grad_weight_adv,
        grad_weight_fm, grad_balancer_ema_decay, use_amp, num_workers, n_steps, warmup_steps,
        evaluation_interval, save_interval,
        in_sample_rate, out_sample_rate, wav_length, segment_length, phone_noise_ratio, vq_topk,
        training_time_vq, floor_noise_level, record_metrics,
        aug_snr_candidates, aug_formant_shift_prob, aug_formant_shift_min, aug_formant_shift_max,
        aug_reverb_prob, aug_lpf_prob, aug_lpf_cutoff_candidates,
        phone_extractor_file, pitch_estimator_file, in_ir_wav_dir, in_noise_wav_dir,
        in_test_wav_dir, pretrained_file,
        pitch_bins, hidden_channels, san, compile_convnext, compile_d4c, compile_discriminator, profile
    ]
    
    reset_button.click(
        reset_inputs,
        outputs=all_outputs
    )

    tensorboard_button.click(
        lambda output_folder_val: gr.Warning(locale_data["tensorboard_alert"]) if not output_folder_val else start_tensorboard(output_folder_val),
        inputs=[output_folder],
        outputs=None,
    )

demo.launch(inbrowser=True)