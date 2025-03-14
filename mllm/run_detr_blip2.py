# 共通のimport
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import requests
import pandas as pd
import math
import matplotlib.pyplot as plt
import japanize_matplotlib
from PIL import Image, ImageDraw
import pickle
import torch
from transformers import BitsAndBytesConfig
from transformers import pipeline
from tqdm import tqdm
import openai
from openai import OpenAI
from src.utils import download_image_from_s3

import torch
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoImageProcessor
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from layout_detr.src.layout_detr import LayoutDetrFeatureExtractor
from layout_detr.src.layout_detr import LayoutDetrForObjectDetection

from PIL import Image
import requests


detor_processor = LayoutDetrFeatureExtractor()
detor_model = LayoutDetrForObjectDetection.from_pretrained('v1-nlabel-1')
detor_model.to('cuda:0')
print()

model_id = '/media/sj-archimedes/data/03_pretrained_model/llm/salesforce/blip2-opt-2.7b'
processor = AutoProcessor.from_pretrained(model_id)
model = Blip2ForConditionalGeneration.from_pretrained(model_id, device_map='auto')

output_dir = "./output_detr_blip2_caption"
target_medias = ['lap', 'fba']


def blip2_caption(image_list, prompt=None):
    num_images = len(image_list)

    caption_result = []
    for img in image_list:
        # 入力の準備
        inputs = processor(img, text=prompt, return_tensors="pt").to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        caption_result.append(generated_text)
    return caption_result


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
            inputs = detor_processor.preprocess(img, return_tensors='pt')
            inputs = {k:v.to(detor_model.device) for k, v in inputs.items()}
            outputs = detor_model(**inputs)
            target_size = torch.Tensor([img.size[::-1]])  # 縦横反転
            postprocessed = detor_processor.post_process_object_detection(outputs, target_sizes=target_size, threshold=0)[0]
            scores = postprocessed['scores'].cpu().detach().numpy()
            boxes = postprocessed['boxes'].cpu().detach().numpy()
            N_BOXES = 10
            ids = scores.argsort()[::-1][:N_BOXES]  # (１つの式にスライスとストライドを同時に使わない. by Effective Python)
            scores = scores[ids]
            boxes = boxes[ids]
            box_images = [img.crop(box) for box in boxes]
            response = blip2_caption(box_images, prompt=None)

        except:
            response = "Error."

        caption_dict[media_hash] = response

        if len(caption_dict) == 10:
            with open(f"{output_dir}/{media}_{i}.pkl", "wb") as w:
                pickle.dump(caption_dict, w)
            caption_dict = {}
print('Done!')

