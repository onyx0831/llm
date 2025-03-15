import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"

import datetime
import gradio as gr
from gradio.mix import Parallel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



line_model_path = "/media/sj-archimedes/data/03_pretrained_model/llm/japanese-large-lm-3.6b-instruction-sft"
line_tokenizer = AutoTokenizer.from_pretrained(line_model_path, use_fast=False)
line_model = AutoModelForCausalLM.from_pretrained(
    line_model_path,
    torch_dtype=torch.float16,
    device_map='auto'
)

elyza_model_name = "/media/sj-archimedes/data/03_pretrained_model/llm/elyza/ELYZA-japanese-Llama-2-7b-fast-instruct"
elyza_tokenizer = AutoTokenizer.from_pretrained(elyza_model_name)
elyza_model = AutoModelForCausalLM.from_pretrained(
    elyza_model_name,
    torch_dtype=torch.float16,
    device_map='auto'
)
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


model1_label = "japanese-large-lm-3.6b-instruction-sft"
model2_label = "ELYZA-japanese-Llama-2-7b-fast-instruct"


def print_log(input_text, output_text, temperature):

    print(datetime.datetime.now())

    print(f"temperature: {temperature}")
    print(f"input: {input_text}")
    print(f"output: {output_text}")
    print('=====')


def chat(model, system_prompt, instruction, template_type, temp=0.7):

    if temp < 0.1:
        temp = 0.1
    elif temp > 1:
        temp = 1


    if template_type == 'line':
        tokenizer = line_tokenizer
        if system_prompt != "":
            instruction = system_prompt + "\n" + instruction
        prompt = f"ユーザー: {instruction}\nシステム: "
        token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    elif template_type == "elyza":
        tokenizer = elyza_tokenizer
        prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
            bos_token=tokenizer.bos_token,
            b_inst=B_INST,
            system=f"{B_SYS}{system_prompt}{E_SYS}",
            prompt=instruction,
            e_inst=E_INST,
        )
        token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=256,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=0,
            repetition_penalty=1.1
        )

    output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True)

    print_log(instruction, output, temp)
    return output

def fn1(system_prompt, instruction, temp):
    return chat(line_model, system_prompt, instruction, 'line', temp)

def fn2(system_prompt, instruction, temp):
    return chat(elyza_model, system_prompt, instruction, 'elyza', temp)


examples = [
    ["日本の観光名所を3つ挙げて。"],
    ["株式会社セプテーニの強みはなんですか？"],
    ["データサイエンティストに必要なスキルを5つ挙げて。"],
    ["美容に関する広告キャッチコピー文を3本提示して。"],
    ["あなたはクリエイティブな広告文作家です。フィットネス の商品のための魅力的なキャッチコピーを考案することが求められています。特に オファー の要素を強調し、体験、初回体験、見学、カウンセリング を明確に示すようにしてください。"],
    ["あなたは経験豊富なマーケティング専門家です。スキンケア 分野の製品をうまく表現することが求められています。機能・成分 を主軸とし、機能、機能、潤い、保湿、乾燥を防ぐ を強調するキャッチコピーを作成してください。"],
    ["あなたは頼れる広告プロデューサーです。スキンケア の製品を市場にアピールするためのキャッチコピーを作成することが必要です。悩み・Before を訴求内容とし、季節悩み、通年・季節の変わり目、通年、季節の変わり目、デイリーユース を強調するものを考えてください。"]
]

demo1= gr.Interface(
    fn1,
    [
        gr.inputs.Textbox(lines=5, label="システムプロンプト"),
        gr.inputs.Textbox(lines=5, label="入力テキスト"),
        gr.inputs.Number(default=0.3, label='Temperature(0.1 ≦ temp ≦ 1 の範囲で入力してください\n小さい値ほど指示に忠実、大きい値ほど多様な表現を出力します。)'),
    ],
    gr.outputs.Textbox(label=model1_label),
    # examples=examples,
    # allow_flagging='never',
    # title="CyberAgent Open-CALM-7b-lora-instruction-tuning",
    # description="""
    #     CyberAgentが公開した大規模言語モデル（68億パラメータ）を会話形式のデータセットで微調整したモデルです。
    #     現状は会話の履歴を保持したやり取りはできず、あくまで1問1答形式でしか答えられません。
    #     社内データは活用していないため、広告に関する知識はほとんど持っていません。
    #     このデモはセプテーニ社内のサーバー上で稼働しています。（入力のログは収集しています。）
    #     """,
)

demo2= gr.Interface(
    fn2,
    [
        gr.inputs.Textbox(lines=5, label="システムプロンプト"),
        gr.inputs.Textbox(lines=5, label="入力テキスト"),
        gr.inputs.Number(default=0.3, label='Temperature(0.1 ≦ temp ≦ 1 の範囲で入力してください\n小さい値ほど指示に忠実、大きい値ほど多様な表現を出力します。)'),
    ],
    gr.outputs.Textbox(label=model2_label),
    examples=examples,
    allow_flagging='never',
    # title="CyberAgent Open-CALM-7b-lora-instruction-tuning",
    # description="""
    #     CyberAgentが公開した大規模言語モデル（68億パラメータ）を会話形式のデータセットで微調整したモデルです。
    #     現状は会話の履歴を保持したやり取りはできず、あくまで1問1答形式でしか答えられません。
    #     社内データは活用していないため、広告に関する知識はほとんど持っていません。
    #     このデモはセプテーニ社内のサーバー上で稼働しています。（入力のログは収集しています。）
    #     """,
)

demo = gr.Parallel(
    demo1,
    demo2,
    examples=examples,
    title="LINE LLM vs ELYZA LLMの検証",
    description="""
        どっちのモデルをベースモデルとすべきか確認したい
        """,)
demo.launch(server_name="0.0.0.0", server_port=8001, share=False)