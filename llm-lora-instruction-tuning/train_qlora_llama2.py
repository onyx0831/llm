import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import argparse
import json
import logging
from logging import config, getLogger  # NOQA
import torch
import torch.nn as nn
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from peft import prepare_model_for_kbit_training
from src.instruct_dataset import InstructDataset


def main(args):
    logger = getLogger(__name__)

    if args.input_file is None:
        input_json_list = datasets.load_dataset("kunishou/databricks-dolly-15k-ja")
        input_json_list = list(input_json_list['train'])
    else:
        with open(args.input_file, 'r') as f:
            input_json_list = json.load(f)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb_config, device_map="auto", pretraining_tp = 1
    )

    train_dataset = InstructDataset(input_json_list, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=args.lora_dropout,
        bias="none",
        fan_in_fan_out=False,
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    logger.info(model.print_trainable_parameters())

    training_args = TrainingArguments(
            output_dir=args.output_dir,
            save_total_limit=1,
            per_device_train_batch_size=args.per_device_train_batch_size,
            num_train_epochs=args.num_train_epochs,
            remove_unused_columns=False,
            logging_steps=args.logging_steps,
            bf16=True,
            dataloader_num_workers=args.dataloader_num_workers,
            report_to="none",
    )

    trainer = Trainer(
            model=model,
            data_collator=collator,
            args=training_args,
            train_dataset=train_dataset,
        )
    model.config.use_cache = False
    

    trainer.train()
    model.save_pretrained(args.output_dir)

    logger.info('Done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="""
        学習データを指定する。List[dict]の形式であり、各dictはinstruction,(input),outputのキーが存在することを前提としている。
        何も指定しなかった場合、datasetsライブラリから[kunishou/databricks-dolly-15k-ja]を持ってくる
        """)
    parser.add_argument("--model_name", type=str, default="cyberagent/open-calm-7b", help="基盤となるLLMのモデル名を指定する。")
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=int, default=0.05)
    parser.add_argument("--target_modules", type=str, default="query_key_value")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--dataloader_num_workers", type=int, default=16)
    parser.add_argument("--output_dir", type=str, help="モデルの保存先を指定する")
    args = parser.parse_args()    
    main(args)