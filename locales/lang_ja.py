# lang_ja.py
lang_data = {
    "title": "Beatrice-Trainer beta2 Unofficial Simple WebUI",

    "input_folder": "[ データフォルダの指定 ]",
    "input_folder_place": " データフォルダのパスを入力してください",
    "input_folder_alert1": "データフォルダのパスが入力されていません",
    "input_folder_alert2": "データフォルダが存在しないか、パスが間違っています",

    "output_folder": "[ 出力先フォルダの指定 ]",
    "output_folder_place": " 出力先フォルダのパスを入力してください",
    "output_folder_alert1": "出力先フォルダのパスが入力されていません",
    "output_folder_alert2": "出力先フォルダが存在しないか、パスが間違っています",

    "args_alert":"未入力の項目があります",

    "config_save_info": "現在の設定でconfig.jsonを書き出します…",
    "checkpoint": "[ 追加学習用チェックポイントファイルの指定 ]",
    "checkpoint_place": "追加学習をさせるチェックポイントファイル名を入力してください",
    "checkpoint_alert": "指定されたcheckpointファイルが無いか、ファイル名が違います",

    "backup_info": "既存の{path}をバックアップします…",
    "rename_info": "{src} を {dest} にリネームします…",
    "train_start": "トレーニングを開始します",
    "addtrain_start": "前回のchekpointからトレーニングを再開します",

    "reset": "リセット",
    "tensorboard": "TensorBoard",
    "tensorboard_alert": "出力フォルダのパスが入力されていません",
    "train": "トレーニング",

    "input_folder_info": "WAVファイルを入れた話者フォルダではなく、ここではその上の階層のデータフォルダのパスを指定してください。\nC:\Beatrice-trainer/datafolder/person01/001.wav　の場合\nC:\Beatrice-trainer/datafolder　までを入力",
    "output_folder_info": "新規学習の時は必ず中身が空のフォルダを指定してください。追加学習・学習再開の時はchekpointファイルがあるフォルダを指定してください。",
    "checkpoint_info": "中断した学習の再開や、追加学習時にファイルを指定してください。\nただし現バージョンでは『指定したステップ数の最後まで学習を完了したファイル』を指定すると開始後にエラーが起きるようです。その場合、学習が完了したファイルではなく『終了のひとつ手前の途中ファイル』を指定するようにしてください。\n例）10000ステップの学習をした場合\ncheckpoint_input_person01_00007000.pt\ncheckpoint_input_person01_00008000.pt ←コレを指定\ncheckpoint_input_person01_00010000.pt",

    "batch_size_info": "１ステップ毎の学習量を指定します。この値が大きいほどVRAMの消費量が増えます。",
    "num_workers_info": "学習データ転送時の並列処理数を指定します。この値が大きいほどメインメモリの消費量が増えます。",
    "n_steps_info": "学習の総ステップ数を指定します。『バッチサイズｘステップ数』が総学習量となります。\n2000ステップごとに途中経過ファイルが保存されます。",

    "in_sample_rate_info": "※変更不可",
    "out_sample_rate_info": "※変更不可"
}
