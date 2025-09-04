# lang_ja.py
lang_data = {
    "title": "Beatrice-Trainer Unofficial Simple WebUI (for 2.0.0-rc.0)",

    "input_folder": "[ 1. データフォルダの指定 ]",
    "input_folder_place": "学習用データセット（話者フォルダ群）が入っているフォルダのパス",
    "input_folder_alert1": "データフォルダのパスが入力されていません",
    "input_folder_alert2": "データフォルダが存在しないか、パスが間違っています",
    "input_folder_info": "音声ファイルが入った話者フォルダ（例: speaker01）の、一つ上の階層のパスを指定します。\n例: C:\data\datafolder の中に speaker01, speaker02… と入っている場合、 C:\data\datafolder までを入力",

    "output_folder": "[ 2. 出力先フォルダの指定 ]",
    "output_folder_place": "学習後のモデルや設定ファイルが出力されるフォルダのパス",
    "output_folder_alert1": "出力先フォルダのパスが入力されていません",
    "output_folder_alert2": "出力先フォルダのパスが不正です",
    "output_folder_info": "学習結果（チェックポイント、設定ファイル等）を保存するフォルダです。存在しない場合は自動で作成されます。\n追加学習の場合は、学習したいチェックポイントが入っているフォルダを指定してください。",

    "checkpoint": "[ 3. (任意) 追加学習用チェックポイント ]",
    "checkpoint_place": "追加学習や学習再開に使用するチェックポイントファイル名（例: checkpoint_xxxx_00010000.pt.gz）",
    "checkpoint_alert": "指定されたcheckpointファイルが無いか、ファイル名が違います",
    "checkpoint_info": "中断した学習の再開や、既存モデルへの追加学習時に指定します。ファイルは出力先フォルダ内にある必要があります。",

    "args_alert":"未入力の項目があります",
    "config_save_info": "現在の設定でconfig.jsonを書き出します…",

    "backup_info": "既存の {path} をバックアップします…",
    "rename_info": "{src} を {dest} (学習再開用ファイル) にリネームします…",
    "train_start": "新規トレーニングを開始します",
    "addtrain_start": "指定されたchekpointからトレーニングを再開します",

    "reset": "設定をリセット",
    "tensorboard": "TensorBoard起動",
    "tensorboard_alert": "出力先フォルダが指定されていません",
    "train": "トレーニング開始",

    "n_steps_info": "学習の総ステップ数です。学習量はおおよそ『バッチサイズ × ステップ数』になります。",
    "batch_size_info": "一度に処理するデータ量です。大きいほどVRAMを多く消費しますが、学習が安定・高速化する傾向があります。",
    "num_workers_info": "データ読み込みの並列処理数です。PCのCPUコア数に応じて設定すると効率が上がりますが、メインメモリの消費量も増えます。",
    "save_interval_info": "指定したステップ数ごとにチェックポイントファイルを保存します。\n基本的には[evaluation_interval_info]と同じ値が良いでしょう",
    "evaluation_interval_info": "指定したステップ数ごとに検証（テスト音声の生成）を行い、品質の目安を確認します。\n基本的には[save_interval]と同じ値が良いでしょう",
    
    "in_sample_rate_info": "入力音声のサンプリングレート（変更不可）",
    "out_sample_rate_info": "出力音声のサンプリングレート（変更不可）",
    "record_metrics_info": "TensorBoardに詳細な損失グラフ等を記録するかどうか。オフにすると僅かに処理負荷が減りますが、学習状況の詳細な分析はできなくなります。",

    # WebUI上の表示の追加
    "advanced_options": "詳細設定",
    "learning_rate_optimizer": "学習率・最適化 (Learning Rate / Optimizer)",
    "loss_weights": "損失の重み (Loss Weights)",
    "augmentation_options": "データ拡張 (Augmentation options)",
    "audio_model": "音声・モデル (Audio / Model)",
    "file_paths": "ファイルパス (File Paths)",
    "performance_debug": "パフォーマンス・デバッグ (Performance / Debug)"
}