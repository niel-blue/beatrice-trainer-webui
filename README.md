## Beatrice Trainer v2 beta.2 Unofficial Simple WebUI Installer

## プログラム概要

本プログラムは、AIボイスチェンジャーVST『Beatrice』の学習キットである『Beatrice Trainer』にWebUIを上乗せしたものです。  
なお、これは『Beatrice Trainer』の公式リリースとは無関係であり、あくまでカスタマイズされた簡易インターフェースを提供するものです。  

『Beatrice』と『Beatrice Trainer』に関する詳細は、以下のリンクをご確認ください：

- [Beatrice (AIボイスチェンジャー VST)](https://huggingface.co/fierce-cats/beatrice)
- [Beatrice Trainer (学習キット)](https://huggingface.co/fierce-cats/beatrice-trainer)


---

### PythonやGitのインストール不要
必要な環境は自動で構築されます（インストールはせずダウンロードして解凍するだけ）。  
ただし環境によってはMicrosoft Visual Studioのインストールが必要。  

---

### 推奨スペック

Beatrice公式から推奨スペックの発表はありません。  
以下は私が独断と偏見で考えたおおよその推奨スペックです

- **OS**：Windows 10または11  
- **CPU**：最近のモデルなら問題ないはずです
- **メモリ**：32GB以上推奨（設定次第で16GBでも動作する可能性あり）
- **グラフィックカード**：NVIDIA RTXシリーズ以上、VRAM 12GB以上（8GBでも動作する可能性あり）
- **ストレージ**：SSD、空き容量20GB以上推奨


### Beatrice Trainer と Webui 両方まとめて導入する方法

1. 本リポジトリをダウンロードし、適切な場所に解凍。

![DL](https://github.com/user-attachments/assets/04a65cb3-163a-40d8-9d35-db803c004ddd)



2. 同梱されている `setup.bat` を実行して、必要な環境をセットアップ。
3. セットアップが完了したら、`run_webui.bat` を実行して、Web UIを起動。



![webui](https://github.com/user-attachments/assets/0d7cd243-edd4-4610-8d47-455bc5df6dbc)



## 現在わかっている仕様
### Beta.2においての追加学習についての制約
- ステップ数がn_stepsのときに学習率が0に近くなるように学習率をスケジューリングしているため、n_stepsで指定したステップ数まで学習が終わったモデルはそれ以上追加学習が行えない。  
- 追加学習をする場合は、学習が終了したファイルよりもひとつ手前の途中保存ファイルをcheckpoint_latest.ptにリネームして学習する必要がある。  
- 途中保存ファイルは2000ステップごとで固定。

### 音声ファイルについて
- 音声ファイルは回数としては均等に使われますが、4秒(デフォルト設定)以上の音声は使うたびに4秒だけ切り抜いて使う仕組みになっているので、長い音声と短い音声が混在していると使われ方に偏りが起きる可能性がある。
- 4秒未満の音声は4秒になるように無音を繋げて使うので、4秒未満の音声はやや非効率の可能性
- サンプリング数は24kHz以上推奨
- 2ch以上の場合は使うたびにランダムなチャンネルを選んで使うので、全部のチャンネルに普通に声が入っていればなんでも良い。
- 事前学習モデルの学習に使ったLibriTTS-Rに含まれないような声はあまり得意ではないかもしれない

