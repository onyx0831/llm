import os
import datetime
import gradio as gr
from gradio.mix import Parallel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from peft import PeftModel, PeftConfig


model_path = "/media/sj-archimedes/data/03_pretrained_model/llm/japanese-large-lm-3.6b-instruction-sft"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_llm = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=False)
model = PeftModel.from_pretrained(base_llm, './output/japanese-large-lm-3.6b-instruction-sft-qlora-instruct-ad')
model.eval()

model_label = "japanese-large-lm-3.6b-instruction-sft-qlora-instruct-ad"


PROMPT_DICT = {
    "prompt_input": (
        "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。"
        "要求を適切に満たす応答を書きなさい。\n\n"
        "### 指示:\n{instruction}\n\n### 入力:{input}\n\n### 応答:"
    ),
    "prompt_no_input": (
        "以下は、タスクを説明する指示です。"
        "要求を適切に満たす応答を書きなさい。\n\n"
        "### 指示:\n{instruction}\n\n### 応答:"
    )
}


def print_log(input_text, output_text, temperature):
    
    print(datetime.datetime.now())
    
    print(f"temperature: {temperature}")
    print(f"input: {input_text}")
    print(f"output: {output_text}")
    print('=====')
    


def chat(model, instruction, temp=0.7, conversation=''):
    
    if temp < 0.1:
        temp = 0.1
    elif temp > 1:
        temp = 1
    
    input_prompt = {
        'instruction': instruction,
        'input': conversation
    }
    
    prompt = ''
    if input_prompt['input'] == '':
        prompt = PROMPT_DICT['prompt_no_input'].format_map(input_prompt)
    else:
        prompt = PROMPT_DICT['prompt_input'].format_map(input_prompt)
    
    inputs = tokenizer(
        prompt,
        padding_side='left',
        return_tensors='pt'
    ).to(model.device)
    
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=temp,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
        )

    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    output = output.split('\n\n')[-1].split(':')[-1]
    
    print_log(instruction, output, temp)
    return output

def fn(instruction, temp):
    return chat(model, instruction, temp)


examples = [
    ["日本の観光名所を3つ挙げて。"],
    ["株式会社セプテーニの強みはなんですか？"],
    ["データサイエンティストに必要なスキルを5つ挙げて。"],
    ["美容に関する広告キャッチコピー文を3本提示して。"],
    ["あなたはクリエイティブな広告文作家です。フィットネス の商品のための魅力的なキャッチコピーを考案することが求められています。特に オファー の要素を強調し、体験、初回体験、見学、カウンセリング を明確に示すようにしてください。"],
    ["あなたは経験豊富なマーケティング専門家です。スキンケア 分野の製品をうまく表現することが求められています。機能・成分 を主軸とし、機能、機能、潤い、保湿、乾燥を防ぐ を強調するキャッチコピーを作成してください。"],
    ["あなたは頼れる広告プロデューサーです。スキンケア の製品を市場にアピールするためのキャッチコピーを作成することが必要です。悩み・Before を訴求内容とし、季節悩み、通年・季節の変わり目、通年、季節の変わり目、デイリーユース を強調するものを考えてください。"]
]

demo = gr.Interface(
    fn,
    [
        gr.inputs.Textbox(lines=5, label="入力テキスト"),
        gr.inputs.Number(default=0.3, label='Temperature(0.1 ≦ temp ≦ 1 の範囲で入力してください\n小さい値ほど指示に忠実、大きい値ほど多様な表現を出力します。)'),
    ],
    gr.outputs.Textbox(label=model_label),
    examples=examples,
    allow_flagging='never',
    title="japanese-large-lm-3.6b-instruction-sft-qlora-instruct-ad",
    description="""
        aaa
        """,
)
demo.launch(server_name="0.0.0.0", server_port=8001, share=False)