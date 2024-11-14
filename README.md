## Beatrice Trainer v2 beta.2 Unofficial Simple WebUI Installer

## プログラム概要

本プログラムは、AIボイスチェンジャーVST『Beatrice』の学習キットである『Beatrice Trainer』にWebUIを上乗せしたものです。
なお、これは『Beatrice Trainer』の公式リリースとは無関係であり、あくまでカスタマイズされた簡易インターフェースを提供するものです。

『Beatrice』と『Beatrice Trainer』に関する詳細は、以下のリンクをご確認ください：

- [Beatrice (AIボイスチェンジャー VST)](https://huggingface.co/fierce-cats/beatrice)
- [Beatrice Trainer (学習キット)](https://huggingface.co/fierce-cats/beatrice-trainer)


---

### PythonやGitのインストール不要
PythonやGitなどの必要な環境は自動で構築されます（インストールはせずダウンロードして解凍するだけ）。
環境によってはMicrosoft Visual Studioのインストールが必要。

---

### 推奨スペック

Beatrice公式から推奨スペックの発表はありません。
以下は私が独断と偏見で考えたおおよその推奨スペックです

- **OS**：Windows 10または11  
- **CPU**：最近のモデルなら問題ないはずです
- **メモリ**：32GB以上推奨（設定次第で16GBでも動作する可能性あり）
- **グラフィックカード**：NVIDIA RTXシリーズ以上、VRAM 12GB以上（8GBでも動作する可能性あり）
- **ストレージ**：SSD、空き容量20GB以上推奨

### 導入方法

1. リポジトリをダウンロードし、適切な場所に解凍。
2. 同梱されている `setup.bat` を実行して、必要な環境をセットアップ。
3. セットアップが完了したら、`run_webui.bat` を実行して、Web UIを起動。

