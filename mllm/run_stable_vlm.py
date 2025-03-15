import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import requests
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from tqdm import tqdm
from transformers import BitsAndBytesConfig
from transformers import pipeline
import pickle
import openai
from openai import OpenAI
from src.utils import download_image_from_s3

import torch
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoImageProcessor

from PIL import Image
import requests

model_id = "/media/sj-archimedes/data/03_pretrained_model/llm/stabilityai/japanese-stable-vlm"
output_dir = "./output_stable_vlm"
target_medias = ['lap', 'fba']

model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
model_kwargs["variant"] = "fp16"
model_kwargs["torch_dtype"] = torch.float16
model_kwargs["device_map"] = "auto"

model = AutoModelForVision2Seq.from_pretrained(model_id, **model_kwargs)
processor = AutoImageProcessor.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = model.eval()

prompt = """
以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

### 指示: 
画像を詳細に述べてください。

### 応答: 
"""

for media in target_medias:
    print(media)
    df = pd.read_pickle(f'/media/sj-archimedes/data/share/OddAI_Library_practice/data08/{media}_trainval_20221001-20231031.pkl')
    df = df.query('creative_type == "image"')
    df['creative_media_url'] = df['creative_media_url'].map(lambda x: x[0])
    df['creative_media_hash'] = df['creative_media_hash'].map(lambda x: x[0])
    df = df[['creative_media_url', 'creative_media_hash']]
    df = df.drop_duplicates()

    caption_dict = {}
    for i, row in tqdm(enumerate(df.itertuples()), total=df.shape[0]):
        s3_url = row.creative_media_url
        media_hash = row.creative_media_hash
        if media_hash in caption_dict:
            continue
        try:
            img = download_image_from_s3(s3_url)
            inputs = processor(images=img, return_tensors="pt")
            text_encoding = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
            inputs.update(text_encoding)
            
            # 推論の実行
            outputs = model.generate(
                **inputs.to(device=model.device),
                do_sample=True,
                num_beams=1,
                max_new_tokens=512,
                min_length=1,
                repetition_penalty=1.5,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        except:
            response = "Error."

        caption_dict[media_hash] = response

        if len(caption_dict) == 10:
            with open(f"{output_dir}/{media}_{i}.pkl", "wb") as w:
                pickle.dump(caption_dict, w)
            caption_dict = {}

print('Done!')
