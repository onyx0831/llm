import os
import datetime
import gradio as gr
from gradio.mix import Parallel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



model_path = "/media/sj-archimedes/data/03_pretrained_model/llm/xwin-lm/Xwin-LM-70B-V0.1"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map='auto'
)
model_label = "XWin-LM"


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

    prompt = f"{system_prompt}\nUSER: {instruction}\n ASSISTANT:"
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=512,
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

def fn(system_prompt, instruction, temp):
    return chat(model, system_prompt, instruction, 'xwin', temp)

examples = [
    ["日本の観光名所を3つ挙げて。"],
    ["株式会社セプテーニの強みはなんですか？"],
    ["データサイエンティストに必要なスキルを5つ挙げて。"],
    ["美容に関する広告キャッチコピー文を3本提示して。"],
    ["あなたはクリエイティブな広告文作家です。フィットネス の商品のための魅力的なキャッチコピーを考案することが求められています。特に オファー の要素を強調し、体験、初回体験、見学、カウンセリング を明確に示すようにしてください。"],
    ["あなたは経験豊富なマーケティング専門家です。スキンケア 分野の製品をうまく表現することが求められています。機能・成分 を主軸とし、機能、機能、潤い、保湿、乾燥を防ぐ を強調するキャッチコピーを作成してください。"],
    ["あなたは頼れる広告プロデューサーです。スキンケア の製品を市場にアピールするためのキャッチコピーを作成することが必要です。悩み・Before を訴求内容とし、季節悩み、通年・季節の変わり目、通年、季節の変わり目、デイリーユース を強調するものを考えてください。"]
]

demo= gr.Interface(
    fn,
    [
        gr.inputs.Textbox(lines=5, label="システムプロンプト"),
        gr.inputs.Textbox(lines=5, label="入力テキスト"),
        gr.inputs.Number(default=0.3, label='Temperature(0.1 ≦ temp ≦ 1 の範囲で入力してください\n小さい値ほど指示に忠実、大きい値ほど多様な表現を出力します。)'),
    ],
    gr.outputs.Textbox(label=model_label),
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

demo = gr.Parallel(
    demo,
    examples=examples,
    title="Xwin-LM-70B",
    description="""
        Open Sournce 最強のLLMの実力や如何に。
        """,)
demo.launch(server_name="0.0.0.0", server_port=8001, share=False)