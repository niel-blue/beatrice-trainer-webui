import gradio as gr
import json
import os
import subprocess
import webbrowser
import locale
import importlib
import shutil

# カレントディレクトリの取得
current_dir = os.getcwd()
default_config_path = os.path.join(current_dir, "assets", "default_config.json")

# 言語設定の読み込み
def load_locale():
    lang_code, _ = locale.getdefaultlocale()
    # 日本語ならlang_ja.py、英語ならlang_en.pyを読み込む
    lang_file = "lang_ja" if lang_code == "ja_JP" else "lang_en"
    # lang_ja.pyやlang_en.pyをimport
    lang_module = importlib.import_module(f"locales.{lang_file}")
    # インポートしたlang_moduleからデータを取得
    return lang_module.lang_data

locale_data = load_locale()


# デフォルトconfig.jsonの読み込み
with open(default_config_path, "r") as f:
    default_config = json.load(f)

# IOフォルダのパスチェック関数
def path_check(input_folder, output_folder, *args):
    if not input_folder:
        gr.Warning(locale_data["input_folder_alert1"])
        return False
    if not os.path.exists(input_folder):
        gr.Warning(locale_data["input_folder_alert2"])
        return False
    if not output_folder:
        gr.Warning(locale_data["output_folder_alert1"])
        return False
    if not os.path.exists(output_folder):
        gr.Warning(locale_data["output_folder_alert2"])
        return False
    for arg in args:
        if arg is None:
            gr.Warning(locale_data["args_alert"])
            return False
    return True

# グローバル変数としてオプションを定義
add_option = ""  # デフォルトは空の文字列

# checkpointファイルの有無チェックとファイルのリネーム関数
def checkpoint_check(output_folder, checkpoint):
    global add_option  # グローバル変数を参照

    if not checkpoint:    # 空の場合はチェックしないでスルー
        return True

    # output_folderに指定されたcheckpointファイルのパスを生成
    checkpoint_path = os.path.join(output_folder, checkpoint)
    
    # checkpointファイルの存在確認
    if not os.path.isfile(checkpoint_path):
        gr.Warning(locale_data["checkpoint_alert"])
        return False
    # checkpoint_latest.ptのパス
    latest_checkpoint_path = os.path.join(output_folder, "checkpoint_latest.pt")
    
    # checkpoint_latest.ptを複製し、元のファイルを削除
    if os.path.isfile(latest_checkpoint_path):
        gr.Info(locale_data["backup_info"].format(path=latest_checkpoint_path))
        shutil.copy2(latest_checkpoint_path, latest_checkpoint_path + ".bak")  
        os.remove(latest_checkpoint_path) 

    # checkpointファイルを複製して、checkpoint_latest.ptとしてリネーム
    gr.Info(locale_data["rename_info"].format(src=checkpoint_path, dest=latest_checkpoint_path))
    shutil.copy2(checkpoint_path, latest_checkpoint_path)

    add_option = "-r"

    return True


# 設定ファイル生成用関数
def generate_config(learning_rate_g, learning_rate_d, min_learning_rate_g, min_learning_rate_d, adam_betas_1, adam_betas_2, adam_eps, batch_size, grad_weight_mel, grad_weight_ap, grad_weight_adv, grad_weight_fm, grad_balancer_ema_decay, use_amp, num_workers, n_steps, warmup_steps, in_sample_rate, out_sample_rate, wav_length, segment_length, phone_extractor_file, pitch_estimator_file, in_ir_wav_dir, in_noise_wav_dir, in_test_wav_dir, pretrained_file, hidden_channels, san, compile_convnext, compile_d4c, compile_discriminator, profile, input_folder,output_folder):
    # 設定ファイルの内容を辞書に格納
    config = {
        "learning_rate_g": float(learning_rate_g),
        "learning_rate_d": float(learning_rate_d),
        "min_learning_rate_g": (min_learning_rate_g),
        "min_learning_rate_d": (min_learning_rate_d),
        "adam_betas": [float(adam_betas_1), float(adam_betas_2)],
        "adam_eps": float(adam_eps),
        "batch_size": int(batch_size),
        "grad_weight_mel": float(grad_weight_mel),
        "grad_weight_ap": float(grad_weight_ap),
        "grad_weight_adv": float(grad_weight_adv),
        "grad_weight_fm": float(grad_weight_fm),
        "grad_balancer_ema_decay": float(grad_balancer_ema_decay),
        "use_amp": bool(use_amp),
        "num_workers": int(num_workers),
        "n_steps": int(n_steps),
        "warmup_steps": int(warmup_steps),
        "in_sample_rate": int(in_sample_rate),
        "out_sample_rate": int(out_sample_rate),
        "wav_length": int(wav_length),
        "segment_length": int(segment_length),
        "phone_extractor_file": phone_extractor_file,
        "pitch_estimator_file": pitch_estimator_file,
        "in_ir_wav_dir": in_ir_wav_dir,
        "in_noise_wav_dir": in_noise_wav_dir,
        "in_test_wav_dir": in_test_wav_dir,
        "pretrained_file": pretrained_file,
        "hidden_channels": int(hidden_channels),
        "san": bool(san),
        "compile_convnext": bool(compile_convnext),
        "compile_d4c": bool(compile_d4c),
        "compile_discriminator": bool(compile_discriminator),
        "profile": bool(profile)
    }
    # 設定ファイルをJSON形式で保存
    with open(output_folder + "/config.json", "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
        gr.Info(locale_data["config_save_info"])
    # 環境変数を設定
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# トレーニングコマンドを実行する関数
def run_training(input_folder, output_folder):
    if add_option == "-r":
        gr.Info(locale_data["addtrain_start"]) 
    else:
        gr.Info(locale_data["train_start"]) 
    # コマンドを作成
    command = [
        "python",
        "beatrice_trainer/__main__.py",
        "-d", input_folder,
        "-o", output_folder,
        "-c", output_folder + "/config.json"
    ]
    # -r オプションがセットされていれば追加
    if add_option == "-r":
        command.append("-r")
    # コマンドの実行
    subprocess.run(command)


# 入力フィールドをリセットする関数
def reset_inputs():
    return [
        "",  # input_folder
        "",  # output_folder
        default_config["batch_size"],
        default_config["num_workers"],
        default_config["n_steps"],
        default_config["warmup_steps"],
        default_config["learning_rate_g"],
        default_config["learning_rate_d"],
        default_config["min_learning_rate_g"],
        default_config["min_learning_rate_d"],
        default_config["adam_betas"][0],  # adam_betas_1
        default_config["adam_betas"][1],  # adam_betas_2
        default_config["adam_eps"],
        default_config["grad_weight_mel"],
        default_config["grad_weight_ap"],
        default_config["grad_weight_adv"],
        default_config["grad_weight_fm"],
        default_config["grad_balancer_ema_decay"],
        default_config["use_amp"],
        default_config["in_sample_rate"],
        default_config["out_sample_rate"],
        default_config["wav_length"],
        default_config["segment_length"],
        default_config["hidden_channels"],
        default_config["in_ir_wav_dir"],
        default_config["in_noise_wav_dir"],
        default_config["in_test_wav_dir"],
        default_config["pretrained_file"],
        default_config["phone_extractor_file"],
        default_config["pitch_estimator_file"],
        default_config["san"],
        default_config["compile_convnext"],
        default_config["compile_d4c"],
        default_config["compile_discriminator"],
        default_config["profile"]
    ]

# TensorBoardを起動する関数
def start_tensorboard(output_folder):
    if output_folder:
        # 以前起動していた Tensorboard プロセスを終了させる
        subprocess.run(["taskkill", "/F", "/IM", "tensorboard.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # 新しい Tensorboard プロセスを起動
        command = ["tensorboard", "--logdir", output_folder]
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        webbrowser.open("http://localhost:6006")

# UI構築
with gr.Blocks() as demo:
    gr.HTML(f"<h1>{locale_data['title']}</h1>")

    with gr.Row():
        with gr.Column():
            input_folder = gr.Textbox(
                label=locale_data["input_folder"],
                placeholder=locale_data["input_folder_place"],
            )
            output_folder = gr.Textbox(
                label=locale_data["output_folder"],
                placeholder=locale_data["output_folder_place"],
            )
            checkpoint = gr.Textbox(
                label=locale_data["checkpoint"],
                placeholder=locale_data["checkpoint_place"],
            )

    with gr.Row():
        batch_size = gr.Number(
            label="Batch Size",
            minimum=1,
            step=1,
            value=default_config["batch_size"],
            interactive=True,
        )
        num_workers = gr.Number(
            label="Num Workers",
            minimum=1,
            step=1,
            value=default_config["num_workers"],
            interactive=True,
        )
        n_steps = gr.Number(
            label="n_steps",
            minimum=1,
            step=1,
            value=default_config["n_steps"],
            interactive=True,
        )
        
    # その他の入力項目
    with gr.Accordion("Advanced options", open=False):

        with gr.Row():
            warmup_steps = gr.Number(label="warmup_steps", value=default_config["warmup_steps"],)
        with gr.Row():
            learning_rate_g = gr.Number(label="Learning Rate G", value=default_config["learning_rate_g"])
            learning_rate_d = gr.Number(label="Learning Rate D", value=default_config["learning_rate_d"])
            min_learning_rate_g = gr.Number(label="Min Learning Rate G", value=default_config["min_learning_rate_g"])
            min_learning_rate_d = gr.Number(label="Min Learning Rate D", value=default_config["min_learning_rate_d"])

        with gr.Row():
            with gr.Row():
                adam_betas_1 = gr.Number(label="Adam Betas", value=default_config["adam_betas"][0])
                adam_betas_2 = gr.Number(label="", value=default_config["adam_betas"][1])
            adam_eps = gr.Number(label="Adam Eps", value=default_config["adam_eps"])

        with gr.Row():
            grad_weight_mel = gr.Number(label="Grad Weight Mel", value=default_config["grad_weight_mel"], step=0.1,precision=2)
            grad_weight_ap = gr.Number(label="Grad Weight AP", value=default_config["grad_weight_ap"], step=0.1,precision=2)
            grad_weight_adv = gr.Number(label="Grad Weight Adv", value=default_config["grad_weight_adv"], step=0.1,precision=2)
            grad_weight_fm = gr.Number(label="Grad Weight FM", value=default_config["grad_weight_fm"], step=0.1,precision=2)
            grad_balancer_ema_decay = gr.Number(label="Grad Balancer EMA Decay", value=default_config["grad_balancer_ema_decay"])

        with gr.Row():
            in_sample_rate = gr.Number(label="In Sample Rate", value=default_config["in_sample_rate"])
            out_sample_rate = gr.Number(label="Out Sample Rate", value=default_config["out_sample_rate"])
            wav_length = gr.Number(label="Wav Length", value=default_config["wav_length"])
            segment_length = gr.Number(label="Segment Length", value=default_config["segment_length"])
            hidden_channels = gr.Number(label="Hidden Channels", value=default_config["hidden_channels"])

        with gr.Column():
                in_ir_wav_dir = gr.Textbox(label="In IR Wav Dir", value=default_config["in_ir_wav_dir"])
                in_noise_wav_dir = gr.Textbox(label="In Noise Wav Dir", value=default_config["in_noise_wav_dir"])
                in_test_wav_dir = gr.Textbox(label="In Test Wav Dir", value=default_config["in_test_wav_dir"])
                pretrained_file = gr.Textbox(label="Pretrained File", value=default_config["pretrained_file"])
                phone_extractor_file = gr.Textbox(label="Phone Extractor File", value=default_config["phone_extractor_file"])
                pitch_estimator_file = gr.Textbox(label="Pitch Estimator File", value=default_config["pitch_estimator_file"])

        with gr.Row():
                san = gr.Checkbox(label="SAN", value=default_config["san"])
                compile_convnext = gr.Checkbox(label="Compile ConvNext", value=default_config["compile_convnext"])
                compile_d4c = gr.Checkbox(label="Compile D4C", value=default_config["compile_d4c"])
                compile_discriminator = gr.Checkbox(label="Compile Discriminator", value=default_config["compile_discriminator"])
                use_amp = gr.Checkbox(label="Use AMP", value=default_config["use_amp"])
                profile = gr.Checkbox(label="Profile", value=default_config["profile"])


    # 各ボタンの構築
    with gr.Row():
        reset_button = gr.Button("Reset")
        tensorboard_button = gr.Button("TensorBoard")
    with gr.Column():
        train_button = gr.Button("Train", variant="primary")

    # トレーニングボタンのクリックイベント
    train_button.click(
        lambda input_folder, output_folder, checkpoint, *args: (
           checkpoint_check(output_folder, checkpoint) and (
                generate_config(*args, input_folder, output_folder) or run_training(input_folder, output_folder)
            ) if path_check(input_folder, output_folder) else None,
        ),
        inputs=[
            input_folder, output_folder, checkpoint,  # checkpointを追加
            learning_rate_g, learning_rate_d, min_learning_rate_g, min_learning_rate_d,
            adam_betas_1, adam_betas_2, adam_eps, batch_size, grad_weight_mel, grad_weight_ap, grad_weight_adv, grad_weight_fm,
            grad_balancer_ema_decay, use_amp, num_workers, n_steps, warmup_steps, in_sample_rate, out_sample_rate,
            wav_length, segment_length, phone_extractor_file, pitch_estimator_file, in_ir_wav_dir, in_noise_wav_dir,
            in_test_wav_dir, pretrained_file, hidden_channels, san, compile_convnext, compile_d4c, compile_discriminator, profile
        ],
        outputs=None,
        show_progress=True
    )


    # リセットボタンのクリックイベント
    reset_button.click(
        reset_inputs,
        outputs=[input_folder, output_folder, batch_size, num_workers, n_steps, warmup_steps, learning_rate_g, learning_rate_d, min_learning_rate_g, min_learning_rate_d, adam_betas_1, adam_betas_2, adam_eps, grad_weight_mel, grad_weight_ap, grad_weight_adv, grad_weight_fm, grad_balancer_ema_decay, use_amp, in_sample_rate, out_sample_rate, wav_length, segment_length, hidden_channels, in_ir_wav_dir, in_noise_wav_dir, in_test_wav_dir, pretrained_file, phone_extractor_file, pitch_estimator_file, san, compile_convnext, compile_d4c, compile_discriminator, profile]
    )

    # Tensorboard ボタンのクリックイベント
    tensorboard_button.click(
        lambda output_folder: gr.Warning(locale_data["tensorboard_alert"]) if not output_folder else None,
        inputs=[output_folder],
        outputs=None,
        show_progress=False
    )

    tensorboard_button.click(
        start_tensorboard,
        inputs=[output_folder],
        outputs=None,
        show_progress=False
    )

    # 説明書きを追加
    input_folder.info = locale_data["input_folder_info"]
    output_folder.info = locale_data["output_folder_info"]
    checkpoint.info = locale_data["checkpoint_info"]
    batch_size.info = locale_data["batch_size_info"]
    num_workers.info = locale_data["num_workers_info"]
    n_steps.info = locale_data["n_steps_info"]
    in_sample_rate.info = locale_data["in_sample_rate_info"]
    out_sample_rate.info = locale_data["out_sample_rate_info"]

demo.launch(inbrowser=True)
