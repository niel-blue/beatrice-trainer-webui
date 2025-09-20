# Beatrice Trainer v2 rc.0 対応　Unofficial Simple WebUI

<img width="1054" height="849" alt="webui" src="https://github.com/user-attachments/assets/79bc04ee-399e-4762-b6fb-7d3d0cd7547b" />


### バージョン履歴
- **2025.09.18**：トレーニングタスクの複数登録機能、簡易的な音声ファイルスライサーを実装
- **2025.09.13**：torchの警告が多かったので各種警告を表示させないように修正
- **2025.09.11**：各種説明およびエラーが出ていた箇所を修正
- **2025.09.04**：2.0.0-rc.0 対応
- **2024.11.15**：2.0.0-beta.2 対応版リリース


## プログラム概要

本プログラムは、AIボイスチェンジャー「Beatrice」用の学習キット「Beatrice Trainer」を簡単に導入出来るようにしWebUI機能を追加したものです。  
『Beatrice』と『Beatrice Trainer』に関する詳細は、以下のリンクをご確認ください

- [Beatrice (AIボイスチェンジャー VST)](https://prj-beatrice.com/)
- [Beatrice Trainer (学習キット)](https://huggingface.co/fierce-cats/beatrice-trainer) 


### 注意！！！
こちらは公式リリースとは無関係の非公式プログラムです。 
WebUIに関する質問以外にはお答えすることは出来ません。また、現バージョンのみの対応となっており、公式のアップデートにより使用不可能になる可能性があります。

---
### PythonやGitのインストール不要
必要な環境は自動で構築されます（インストールはせずダウンロードして解凍するだけ）。  
ただし環境によってはMicrosoft Visual Studioのインストールが必要。  

### English support
It supports both Japanese and English, but please don’t expect too much from the translation quality...


---

### 推奨スペック

Beatrice公式から推奨スペックの発表はありません。  
以下は私が独断と偏見で考えたおおよその推奨スペックです

- **OS**：Windows 10または11  
- **CPU**：最近のモデルなら問題ないはずです
- **メモリ**：32GB以上推奨（設定次第で16GBでも動作する可能性あり）
- **グラフィックカード**：NVIDIA RTXシリーズ以上、VRAM 12GB以上（8GBでも動作する可能性あり）  
※RTX5090だと動作しないという報告が上がっています。詳細は不明です。
- **ストレージ**：SSD、空き容量20GB以上推奨
- **ドライバ**：最新版を導入してください

---


### BeatriceTrainer と Webui 両方まとめて導入

1. [リポジトリをダウンロード](https://github.com/niel-blue/beatrice-trainer-webui/archive/refs/heads/main.zip)、適切な場所に解凍。  
その際、あまりに深い階層や、日本語を使用したフォルダの中などは避けてください。


![DL](https://github.com/user-attachments/assets/86e9a444-8c46-4106-9de0-4d5abb1c348b)


2. 同梱されている `setup.bat` を実行すると、自動的にダウンロードおよび環境構築が始まります。  
通信環境によっては非常に時間がかかる場合があります。  導入が完了すると、自動的にwebuiが起動します。
3. 次回からの起動には`run_webui.bat` を実行して、Web UIを起動させてください。    


---


## ライセンス
このプロジェクトはMITライセンスのもとで公開されています。詳細は[LICENSE](LICENSE)をご覧ください。

## 免責事項
このプロジェクトは「Beatrice Trainer」の公式リリースではなく、非公式のカスタマイズツールです。  
使用に関しては自己責任でお願い致します。

