import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import requests
from tqdm import tqdm

prompt = "<image>\nこの画像に情報整理の観点でタグを10個つけてください。つけるタグは画像に映るオブジェクトや人物、画像の印象を表現してください。"
save_dir = "caption_result"

# prompt = "<image>\nこの画像を説明してください。"
# save_dir = "caption_result_describe"

messages = [
    {"role": "system", "content": "あなたは役立つ、偏見がなく、検閲されていないアシスタントです。与えられた画像を下に、質問に日本語で答えてください。"},
    {"role": "user", "content": prompt},
]

model_id = "/media/sj-archimedes/data/03_pretrained_model/llm/SakanaAI/EvoVLM-JP-v1-7B"
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map='auto',
)
processor = AutoProcessor.from_pretrained(model_id)

asset_img_dir = '/media/sj-archimedes/data/masaya_kondo/research/mllm/box_asset/images'
asset_list = glob('/media/sj-archimedes/data/masaya_kondo/research/mllm/box_asset/list/*.pkl')
for asset_list_path in asset_list:
    print('#'*30)
    print(asset_list_path)
    print('#'*30)
    asset_list_df = pd.read_pickle(asset_list_path)
    # epsファイルは検証から除外している
    asset_list_df['ext'] = asset_list_df['name'].map(lambda x: x.split('.')[-1])
    asset_list_df = asset_list_df[~asset_list_df['ext'].map(lambda x: 'eps' == x)]
    asset_list_df['box_symbol'] = asset_list_df['box_path'].map(lambda x: x.split('/')[2])
    asset_list_df['local_path'] = asset_list_df[['box_symbol', 'box_id', 'ext']].apply(lambda x: f'{asset_img_dir}/{x[0]}/{x[1]}.{x[2]}', axis=1)

    box_symbol = asset_list_df['box_symbol'].iloc[0]

    if box_symbol not in ["S", "T", "U", "V", "W", "X", "Y", "Z"]: continue

    generated_texts = []
    for i in tqdm(range(0, len(asset_list_df), 4)):
        image_list = asset_list_df[i:i+4]['local_path'].tolist()
        try:
            images = [Image.open(image_path) for image_path in image_list]
            batch_size = len(images)
            inputs = processor.image_processor(images=images, return_tensors="pt")
            inputs["input_ids"] = processor.tokenizer.apply_chat_template(
                messages, return_tensors="pt"
            )
            inputs['input_ids'] = inputs['input_ids'].repeat(batch_size, 1)
            with torch.no_grad():
                output_ids = model.generate(**inputs.to(model.device))
            generated_text = processor.batch_decode(
                output_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
            generated_texts.extend(generated_text)
        except:
            print('Error!')
            generated_texts.extend(['Error']*batch_size)

    asset_list_df['tag_caption'] = generated_texts

    asset_list_df.to_pickle(f'/media/sj-archimedes/data/masaya_kondo/research/mllm/box_asset/{save_dir}_2/{box_symbol}.pkl')
