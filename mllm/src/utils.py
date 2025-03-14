import warnings
import logging
from typing import Union, List, Mapping
import os
import io
import re
import urllib
from urllib.parse import urlparse
import pandas as pd
from tqdm import tqdm
from PIL import Image

def download_image_from_s3(s3_url: str, retry_time: int = 5) -> Image.Image:
    """
    s3_urlから画像を取得する。
    Args:
        s3_url: 画像の格納先URL。以下の全パターンに対応
            1. https://s3downloader.*
            2. https://sep-cruc.s3.ap-northeast-1.amazonaws.com.*
            3. https://tech-asset-storage-prod.s3-ap-northeast-1.amazonaws.com.*
            4. https://prod-odd-ai-terminal-assets.s3.ap-northeast-1.amazonaws.com.*
            5. s3://*  (仕様外だが一応実装)
        retry_time:(int) 読み込み失敗した時、この回数だけ繰り返します。
    Return:
        画像オブジェクト or None
        s3_url: 画像の格納先URL
    """
    parsed = urlparse(s3_url)
    # s3:// の場合
    if parsed.scheme == "s3":
        warnings.warn(
            "s3://... format url is out of specification, not fully maintained."
        )
        import boto3

        bucket = boto3.resource("s3").Bucket(parsed.netloc)
        s3_bin = io.BytesIO()
        try:
            bucket.download_fileobj(parsed.path[1:], s3_bin)
        except Exception as e:
            raise ValueError(f"s3_path not found. {e}")
        s3_bin.seek(0)
        img = Image.open(s3_bin).convert("RGB")
        return img

    # https:// の場合
    # viewerの場合、pathを修正。
    if re.match("s3downloader-viewer", parsed.netloc):
        s3_url = re.sub(r"s3downloader-viewer.", "s3downloader.", s3_url)
        s3_url = re.sub(r"/view\?", "/file?", s3_url)

    # pathのファイル（or ファイルもどき）を読みs3_url
    # （データが欠落していても、本来のサイズに128値埋めして読み込む）
    try:
        s3_bin = io.BytesIO(urllib.request.urlopen(s3_url).read())
        img = Image.open(s3_bin).convert("RGB")
    except Exception as e:
        # 再raiseする。より具体的な対エラーアクションは未定
        logger.error(e)
        logger.error("error url:", s3_url, f"\n{retry_time} trials left")
        if retry_time <= 0:
            raise
        else:
            logger.info("Retry downloading...")
            return _download_image_from_s3(s3_url, retry_time - 1)
    return img