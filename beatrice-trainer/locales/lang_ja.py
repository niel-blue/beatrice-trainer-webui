# VERSION = "25.09.18"
# lang_ja.py
lang_data = {
    "LNG_TITLE": "Beatrice-Trainer Unofficial Simple WebUI (for 2.0.0-rc.0)",
    "LNG_TRAIN_DESC": "入出力フォルダ・各種パラメータを設定し、[タスクをキューに登録]ボタンでタスクを登録してください（※複数可）。[トレーニング開始]でタスクが順番に実行されます。",
    
    "LNG_STOP_SUCCESS_MESSAGE": "トレーニングプロセスを停止しました。キューをクリアします。",
    "LNG_STOP_NO_PROCESS_MESSAGE": "実行中のトレーニングプロセスはありません。キューをクリアします。",
    "LNG_RESUME_WARNING": "指定されたn_stepsの値 ({n_steps_val}) が、既存のconfig.jsonのステップ数 ({existing_steps}) 以下です。継続して学習を行うには、より大きな値を指定してください。",
    "LNG_CONFIG_LOAD_ERROR": "config.jsonの読み込み中に問題が発生しました: {e}",
    "LNG_CONFIG_NOT_FOUND_WARNING": "追加学習用のconfig.jsonが見つかりませんでした。新規学習として扱います。",
    "LNG_LATEST_CHECKPOINT_BACKUP": "既存の最新チェックポイントをバックアップしました: {backup_path}",
    
    "LNG_NO_AUDIO_FILES_IN_ROOT_ALERT": "指定されたフォルダの直下に音声ファイルがあります。話者ごとにサブフォルダを作成し、その中に音声を配置してください。",
    "LNG_NO_AUDIO_FILES_IN_SUBFOLDERS_ALERT": "音声ファイルが見つかりません。話者フォルダ内に音声を配置しているか確認してください。",

    # トレーニングタブ
    "LNG_TAB_TRAIN": "トレーニング",

    "LNG_INPUT_FOLDER": "[データフォルダの指定]",
    "LNG_INPUT_FOLDER_INFO": "データフォルダの中に話者名フォルダを作成し、音声ファイルを配置してください。例: C:\\data\\datafolder\speaker01\\001.wav",
    "LNG_INPUT_FOLDER_PLACE": "例: C:\\data\\datafolder の中に speaker01, speaker02… がある場合、 C:\\data\\datafolder までのパスを指定。",
    "LNG_INPUT_FOLDER_ALERT": "データフォルダのパスが入力されていないか、パスが間違っています",

    "LNG_OUTPUT_FOLDER": "[出力先フォルダの指定]",
    "LNG_OUTPUT_FOLDER_PLACE": "新規学習の場合は任意のフォルダパスを指定。学習再開や追加学習の場合は対象のチェックポイントが入っているフォルダを指定。",
    "LNG_OUTPUT_FOLDER_ALERT": "出力先フォルダのパスが入力されていないか、パスが間違っています",

    "LNG_CHECKPOINT": "[追加学習時のチェックポイント指定（任意）]",
    "LNG_CHECKPOINT_PLACE": "checkpoint_latest.pt.gz以外のファイルから学習を再開させたい時のみファイル名を指定。例: checkpoint_100000_00001000.pt.gz",

    "LNG_CONFIG_SAVE_INFO": "config.json が生成されました。",
    "LNG_TENSORBOARD_ALERT": "TensorBoardを起動するには、まず出力フォルダを指定してください。",
    
    "LNG_BASIC_TRAINING": "基本トレーニング設定",
    
    "LNG_N_STEPS_INFO": "トレーニングの総ステップ数。品質にも大きく影響。増やすと学習時間も増加する。",
    "LNG_BATCH_SIZE_INFO": "一度に処理する学習データ数。増やすと安定性が向上するなど学習速度や品質に大きく影響するがVRAM使用量も増加。",
    "LNG_NUM_WORKERS_INFO": "データ読み込みを並列処理するスレッド数。CPUコア数に合わせると効率が上がりますがメモリ消費も増加。",
    "LNG_SAVE_INTERVAL_INFO": "指定ステップごとにチェックポイントファイルを保存します。",
    "LNG_EVALUATION_INTERVAL_INFO": "指定ステップごとにテスト音声の生成を行い、学習の進み具合を検証。",

    "LNG_ADVANCED_OPTIONS": "詳細設定",
    
    "LNG_LEARNING_RATE_OPTIMIZER": "学習率・オプティマイザ",
    "LNG_LOSS_WEIGHTS": "損失関数の重み",
    "LNG_AUGMENTATION_OPTIONS": "データ拡張オプション",
    "LNG_AUDIO_MODEL": "オーディオ・モデルパラメータ",
    "LNG_FILE_PATHS": "ファイルパス",
    "LNG_PERFORMANCE_DEBUG": "パフォーマンス・デバッグ",

    "LNG_IN_SAMPLE_RATE_INFO": "変更不可。",
    "LNG_OUT_SAMPLE_RATE_INFO": "変更不可。",
    
    "LNG_TASK_MONITOR": "### [タスクモニター]\n",
    "LNG_TASK_MONITOR_EMPTY": "現在キューは空です。タスクを追加してください。",
    "LNG_QUEUE_TASK_ADDED": "タスクをキューに登録しました: {output_folder}",
    "LNG_QUEUE_TASK_RESUMED": "追加学習としてキューに登録しました: {output_folder}",
    "LNG_QUEUE_TASK_NEW": "新規学習としてキューに登録しました: {output_folder}",
    "LNG_QUEUE_STATUS_PENDING": "[待機中◼]",
    "LNG_QUEUE_STATUS_IN_PROGRESS": "[実行中▶]",
    "LNG_QUEUE_STATUS_COMPLETED": "[完了済✅]",
    "LNG_QUEUE_STATUS_ERROR": "[エラー❌]",
    "LNG_QUEUE_STATUS_STOPPED": "[中止🛑]",
    "LNG_QUEUE_START_INFO": "キューからタスクを開始します: {output_folder}",
    "LNG_USER_TERMINATED_MESSAGE": "タスクはユーザーにより中断されました。",
    "LNG_ALL_TASKS_COMPLETED_MESSAGE": "キュー内の全タスクが完了しました。",

    "LNG_ADD_TASK_BUTTON": "タスクをキューに登録",
    "LNG_START_TRAINING_BUTTON": "トレーニング開始",
    "LNG_STOP_TRAINING_BUTTON": "中断/キューのクリア",
    "LNG_TENSORBOARD_BUTTON": "TensorBoardを起動",

    # データセット前処理タブ
    "LNG_TAB_DATASET_PROCESSING": "データセット前処理" ,
    "LNG_DATASET_PROCESSING_DESC": "音声ファイル（wav、ogg、mp3、flacなど）を指定した長さに分割し、wavやflac形式に変換して保存出来ます。",
    
    "LNG_INPUT_DIR_PREP_LABEL": "[入力フォルダの指定]",
    "LNG_INPUT_DIR_PREP_PLACE": "前処理したい音声ファイル（複数可）が入っているフォルダパスを指定。",
    "LNG_OUTPUT_DIR_PREP_LABEL": "[出力先フォルダの指定（任意）]",
    "LNG_OUTPUT_DIR_PREP_PLACE": "処理された音声ファイルの保存先を指定。空欄の場合は入力フォルダ内に自動的にフォルダを作成して出力します。",
    "LNG_SEGMENT_DURATION_LABEL": "音声の分割長さ（秒）",
    "LNG_SEGMENT_DURATION_INFO": "音声を指定した長さで分割します。（※4秒 推奨）",
    "LNG_OUTPUT_SAMPLERATE_LABEL": "サンプリングレート",
    "LNG_OUTPUT_SAMPLERATE_INFO": "出力する音声のサンプリングレートを選択（※16000Hz 推奨）",
    "LNG_OUTPUT_FORMAT_LABEL": "出力形式を選択",
    "LNG_OUTPUT_FORMAT_INFO": "保存するファイル形式を選択してください。",
    "LNG_SILENCE_REMOVAL_OPTIONS": "無音部分削除オプション",
    "LNG_SILENCE_REMOVAL_OPTIONS_INFO": "無音部分を自動で除去するかどうかを制御します。",
    "LNG_SILENCE_THRESHOLD_LABEL": "無音の閾値（dBFS）",
    "LNG_SILENCE_THRESHOLD_INFO": "この値より小さい音量を無音と見なします。",
    "LNG_MIN_SILENCE_DURATION_LABEL": "最小無音時間（ミリ秒）",
    "LNG_MIN_SILENCE_DURATION_INFO": "この時間よりも長い無音部分を除去します。",

    "LNG_STATUS_WAITING": "### 状況: 待機中",
    "LNG_ERROR_NO_INPUT_FOLDER": "入力フォルダを指定してください。",
    "LNG_WARNING_NO_AUDIO_FILES": "指定されたフォルダに音声ファイルが見つかりませんでした。",
    "LNG_ERROR_OUTPUT_FOLDER_CREATION": "出力フォルダ '{output_dir}' の作成に失敗しました。<br>{e}",
    "LNG_COMPLETE_WITH_FAILURES": "処理が完了しました。成功: {processed_count}、失敗: {len(failed_files)}。<br>失敗ファイル:<br>{failed_message}",
    "LNG_COMPLETE_SUCCESS": "処理が完了しました。{processed_count} ファイルを処理しました。スキップ: {skipped_count}。出力先: {output_dir}",

    "LNG_SPLIT_BUTTON": "音声ファイル書き出し"

}