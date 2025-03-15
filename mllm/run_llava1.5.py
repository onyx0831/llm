import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import requests
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import pickle
from tqdm import tqdm
from transformers import BitsAndBytesConfig
from transformers import pipeline
import japanize_matplotlib
import openai
from openai import OpenAI
from src.utils import download_image_from_s3

model_id = "/media/sj-archimedes/data/03_pretrained_model/llm/llava-hf/llava-1.5-7b-hf"
output_dir = "./output_appeal_caption"
target_medias = ['lap', 'fba']

pipe = pipeline(
    "image-to-text",
    model=model_id,
    device_map='auto',
)

# prompt = "USER: <image>\nPlease describe this image.\nASSISTANT:"
prompt = "USER: <image>\nPlease explain in detail how this advertisement image has been designed to appear attractive to consumers.\nASSISTANT:"

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
            outputs = pipe(
                img,
                prompt=prompt,
                generate_kwargs={"max_new_tokens": 512}
            )
            response = outputs[0]['generated_text'].split('\nASSISTANT: ')[1]
        except:
            response = "Error."

        caption_dict[media_hash] = response

        if len(caption_dict) == 10:
            with open(f"{output_dir}/{media}_{i}.pkl", "wb") as w:
                pickle.dump(caption_dict, w)
            caption_dict = {}

print('Done!')

