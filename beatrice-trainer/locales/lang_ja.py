# lang_ja.py
lang_data = {
    "title": "Beatrice-Trainer Unofficial Simple WebUI (for 2.0.0-rc.0)",

    "input_folder": "[ 1. データフォルダの指定 ]",
    "input_folder_place": "学習用データセット（話者フォルダ群）が入っているフォルダのパス",
    "input_folder_alert1": "データフォルダのパスが入力されていません",
    "input_folder_alert2": "データフォルダが存在しないか、パスが間違っています",
    "input_folder_info": "音声ファイルが入った話者フォルダ（例: speaker01）の一つ上の階層を指定してください。\n例: C:\\data\\datafolder の中に speaker01, speaker02… がある場合、 C:\\data\\datafolder までを入力してください。",

    "output_folder": "[ 2. 出力先フォルダの指定 ]",
    "output_folder_place": "学習後のモデルや設定ファイルが出力されるフォルダのパス",
    "output_folder_alert1": "出力先フォルダのパスが入力されていません",
    "output_folder_alert2": "出力先フォルダのパスが不正です",
    "output_folder_info": "学習結果（チェックポイント、設定ファイル等）を保存するフォルダです。存在しない場合は自動で作成されます。\n追加学習の場合は、対象のチェックポイントが入っているフォルダを指定してください。",

    "checkpoint": "[ 3. (任意) 追加学習用チェックポイント ]",
    "checkpoint_place": "追加学習や学習再開に使用するチェックポイントファイル名（例: checkpoint_xxxx_00010000.pt.gz）",
    "checkpoint_alert": "指定されたcheckpointファイルが存在しないか、ファイル名が間違っています",
    "checkpoint_info": "中断した学習の再開や、既存モデルへの追加学習時に指定します。ファイルは出力先フォルダ内にある必要があります。",

    "args_alert":"未入力の項目があります",
    "config_save_info": "現在の設定で config.json を書き出します…",

    "backup_info": "既存の {path} をバックアップします…",
    "rename_info": "{src} を {dest} (学習再開用ファイル) にリネームします…",
    "train_start": "新規トレーニングを開始します",
    "addtrain_start": "指定されたチェックポイントからトレーニングを再開します",

    "reset": "設定をリセット",
    "tensorboard": "TensorBoard起動",
    "tensorboard_alert": "出力先フォルダが指定されていません",
    "train": "トレーニング開始",

    "n_steps_info": "学習の総ステップ数です。\n（ステップ数×バッチサイズ）が総学習量の目安となり、品質にも影響します。",
    "batch_size_info": "一度に処理するデータ数です。大きくするとVRAM消費が増えますが、安定性が向上するなど、学習速度や品質に大きく影響します。",
    "num_workers_info": "データ読み込みを並列処理するスレッド数です。CPUコア数に合わせると効率が上がりますが、メモリ消費も増えます。",
    "save_interval_info": "指定ステップごとにチェックポイントファイルを保存します。\n通常は [evaluation_interval] と同じ値にするのが良いでしょう。",
    "evaluation_interval_info": "指定ステップごとに検証（テスト音声の生成）を行い、学習の進み具合を確認します。\n通常は [save_interval_info] と同じ値にするのが良いでしょう。",
    
    "in_sample_rate_info": "入力音声のサンプリングレート（固定値）",
    "out_sample_rate_info": "出力音声のサンプリングレート（固定値）",
    "record_metrics_info": "TensorBoardに詳細な損失や指標を記録します。無効にすると処理は僅かに軽くなりますが、学習の分析はできなくなります。",

    "use_amp_info": "AMP（Automatic Mixed Precision、自動混合精度）を使用します。半精度演算を導入することでVRAM使用量を削減し、学習を高速化できます。ただし環境によっては不安定になる可能性があります。",
    "san_info": "SAN（Self-Attention Network）ベースの識別器を使用します。生成音声をより高品質に判別できますが、計算コストが増加します。",
    "profile_info": "学習処理を詳細に記録し、ボトルネック解析やデバッグに役立てます。ただし有効化すると処理速度は低下します。",

    "basic_training": "基本トレーニング設定",
    "advanced_options": "詳細設定",
    "learning_rate_optimizer": "学習率・最適化",
    "loss_weights": "損失の重み",
    "augmentation_options": "データ拡張",
    "audio_model": "音声・モデル",
    "file_paths": "ファイルパス",
    "performance_debug": "パフォーマンス・デバッグ"
}
