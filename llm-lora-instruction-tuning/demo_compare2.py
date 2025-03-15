import os
import datetime
import gradio as gr
from gradio.mix import Parallel
from gradio.components import Textbox, Number

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from peft import PeftModel, PeftConfig


model_path = "/media/sj-archimedes/data/03_pretrained_model/llm/japanese-large-lm-3.6b-instruction-sft"
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=False,
    padding_side='left',
    local_files_only=True,
)

model1 = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)

model2 = AutoModelForCausalLM.from_pretrained(
    '/media/sj-archimedes/data/masaya_kondo/sft_llm/japanese-large-lm-3.6b-riken-instruct-sft',
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)


def print_log(model_name, input_text, output_text):
    
    print(datetime.datetime.now())    
    print(f"model: {model_name}")
    print(f"input: {input_text}")
    print(f"output: {output_text}")
    print('=====')
    


def chat(model, input_text, model_name):

    if model_name == 'riken':
        text = f"### USER: {input_text} \n### ASSISTANT: "
    else:
        text = f"ユーザー: {input_text}\nシステム: "

    with torch.no_grad():
    
        inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False).to(model.device)
        token = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample = True,
            temperature = 0.7,
            top_p = 0.9,
            top_k = 0,
            repetition_penalty = 1.1,
            num_beams = 1,
            num_return_sequences=1
        )
    output = tokenizer.decode(token[0], skip_special_tokens=True)    
    print_log(model_name, input_text, output)

    if model_name == 'riken':
        output = output.split('### ASSISTANT: ')[1]
    else:
        output = output.split('\nシステム: ')[1]
    
    return output

def fn1(input_text):
    return chat(model1, input_text, 'oasst')

def fn2(input_text):
    return chat(model2, input_text, 'riken')


examples = [
    ["日本の観光名所を3つ挙げて。"],
    ["株式会社セプテーニの強みはなんですか？"],
    ["データサイエンティストに必要なスキルを5つ挙げて。"],
    ["美容に関する広告キャッチコピー文を3本提示して。"],
]

demo1 = gr.Interface(
    fn1,
    [
        Textbox(lines=5, label="入力テキスト"),
    ],
    Textbox(label='oasst'),
)

demo2 = gr.Interface(
    fn2,
    [
        Textbox(lines=5, label="入力テキスト"),
    ],
    Textbox(label="riken"),
)

demo = gr.Parallel(
    demo1,
    demo2,
    examples=examples,
    title="LINE LLMのSFTデータ検証",
    description="""
        - oasst: LINEがOASSTデータでSFTしたモデル
        - riken: LINE LLMをベースにRikenの日本語インストラクションデータでSFTしたモデル
        """,)
demo.launch(server_name="0.0.0.0", server_port=8001, share=False)