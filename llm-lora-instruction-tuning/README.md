# LLM-LoRA-instruction-tuning

## 主要ライブラリのバージョン
以下のバージョン以外での動作は未確認

```sh
datasets                   2.12.0
peft                       0.2.0
torch                      1.13.1
transformers               4.27.1
```

## 学習コマンドの例

```sh
python -u train.py \
    --input_file ./input/dolly_ja.json \
    --model_name cyberagent/open-calm-7b \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --dataloader_num_workers 16 \
    --output_dir ./output/lora
```

```sh
python -u train_qlora.py \
    --input_file ./input/dolly_and_appeal.json \
    --model_name /media/sj-archimedes/data/03_pretrained_model/llm/japanese-large-lm-3.6b-instruction-sft \
    --per_device_train_batch_size 8 \
    --num_train_epochs 3 \
    --dataloader_num_workers 16 \
    --output_dir ./output/japanese-large-lm-3.6b-instruction-sft-qlora-instruct-dolly-ad \
    --r 8 \
    --lora_alpha 16
```


## デモの起動

`demo.py`や`demo_compare.py`でモデルのデモをgradioで行うことができます。

コード内で適宜学習済モデルの読み込み先を指定してください。

`open-calm-7b`を前提としています。


- 1つのモデルのデモを行う場合

```sh
python -u demo.py
```

- 2つのモデルのデモを同時に行いたい場合

```sh
python -u demo_compare.py
```


# 連携事項
- LLama-2-70b-chatモデルのdolly-jaデータセットでQLoRA学習したadapterのパラメータは以下に格納しています。

```
/media/sj-archimedes/data/03_pretrained_model/llm/llama2-70b-chat-qlora-dolly-ja
```