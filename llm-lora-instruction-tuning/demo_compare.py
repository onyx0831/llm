import os
import datetime
import gradio as gr
from gradio.components import Textbox, Number
from gradio.mix import Parallel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from src.instruct_template import prompt_template_dict

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_base_path = "/media/sj-archimedes/data/03_pretrained_model/llm/stabilityai"
model_list = [
    "japanese-stablelm-3b-4e1t-base",
    "japanese-stablelm-3b-4e1t-instruct",
    "japanese-stablelm-base-gamma-7b",
    "japanese-stablelm-instruct-gamma-7b"
]

tokenizer_dict = {
    model_name: AutoTokenizer.from_pretrained(
        os.path.join(model_base_path, model_name),
        local_files_only=True
    ) for model_name in model_list
}

model_dict = {}
model_dict = {
    model_name: AutoModelForCausalLM.from_pretrained(
        os.path.join(model_base_path, model_name),
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        # quantization_config=bnb_config
    ) for model_name in model_list if "4e1t" in model_name
}
model_dict.update({
    model_name: AutoModelForCausalLM.from_pretrained(
        os.path.join(model_base_path, model_name),
        device_map="auto",
        torch_dtype=torch.float16,
        # quantization_config=bnb_config
    ) for model_name in model_list if "gamma" in model_name
})

def build_prompt(instruction, inputs="", sep="\n\n### "):
    sys_msg = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"
    p = sys_msg
    roles = ["指示", "応答"]
    msgs = [": \n" + instruction, ": \n"]
    if inputs:
        roles.insert(1, "入力")
        msgs.insert(1, ": \n" + inputs)
    for role, msg in zip(roles, msgs):
        p += sep + role + msg
    return p


def print_log(model_name, instruction, inputs, prompt, output_text, temperature):

    print(datetime.datetime.now())
    print(f"model_name: {model_name}")
    print(f"temperature: {temperature}")
    print(f"instruction: {instruction}")
    print(f"inputs: {inputs}")
    print(f"prompt: {prompt}")
    print(f"output: {output_text}")
    print('=====')


def echo(instruction, inputs, temp):
    user_inputs = {
        'instruction': instruction,
        'inputs': inputs
    }
    prompt = build_prompt(**user_inputs)
    return prompt


def chat(model, instruction, inputs, model_name, temp=0.7):

    if temp < 0.1:
        temp = 0.1
    elif temp > 1:
        temp = 1

    user_inputs = {
        'instruction': instruction,
        'inputs': inputs
    }

    prompt = build_prompt(**user_inputs)

    input_ids = tokenizer_dict[model_name].encode(
        prompt,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)
    # inputs = {k:v[:, :-1] for k, v in inputs.items()}

    with torch.no_grad():
        tokens = model.generate(
            input_ids,
            max_new_tokens=256,
            temperature=temp,
            top_p=0.95,
            repetition_penalty=1.1,
            do_sample=True,
        )

    output = tokenizer_dict[model_name].decode(tokens[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    # output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    # output = output.split('\n\n')[-1].split(':')[-1]

    print_log(model_name, instruction, inputs, prompt, output, temp)
    return output

def fn1(instruction, inputs, temp):
    return chat(
        model_dict["japanese-stablelm-3b-4e1t-base"],
        instruction,
        inputs,
        "japanese-stablelm-3b-4e1t-base",
        temp
    )

def fn2(instruction, inputs, temp):
    return chat(
        model_dict["japanese-stablelm-3b-4e1t-instruct"],
        instruction,
        inputs,
        "japanese-stablelm-3b-4e1t-instruct",
        temp
    )

def fn3(instruction, inputs, temp):
    return chat(
        model_dict["japanese-stablelm-base-gamma-7b"],
        instruction,
        inputs,
        "japanese-stablelm-base-gamma-7b",
        temp
    )

def fn4(instruction, inputs, temp):
    return chat(
        model_dict["japanese-stablelm-instruct-gamma-7b"],
        instruction,
        inputs,
        "japanese-stablelm-instruct-gamma-7b",
        temp
    )

examples = [
    ["日本の観光名所を3つ挙げて。"],
    ["株式会社セプテーニの強みはなんですか？"],
    ["データサイエンティストに必要なスキルを5つ挙げて。"],
    ["美容に関する広告キャッチコピー文を3本提示して。"],
]

demo_echo = gr.Interface(
    echo,
    [
        Textbox(lines=5, label="指示"),
        Textbox(lines=5, label="入力"),
        gr.Slider(
            label='Temperature',
            minimum=0.1,
            maximum=1.0,
            step=0.1,
            value=0.7,
        )
    ],
    Textbox(label="LLMに入力されたテキスト"),
)

demo1 = gr.Interface(
    fn1,
    [
        Textbox(lines=5, label="指示"),
        Textbox(lines=5, label="入力"),
        gr.Slider(
            label='Temperature',
            minimum=0.1,
            maximum=1.0,
            step=0.1,
            value=0.7,
        )
    ],
    Textbox(label="japanese-stablelm-3b-4e1t-base"),
)

demo2 = gr.Interface(
    fn2,
    [
        Textbox(lines=5, label="指示"),
        Textbox(lines=5, label="入力"),
        gr.Slider(
            label='Temperature',
            minimum=0.1,
            maximum=1.0,
            step=0.1,
            value=0.7,
        )
    ],
    Textbox(label="japanese-stablelm-3b-4e1t-instruct"),
)

demo3 = gr.Interface(
    fn3,
    [
        Textbox(lines=5, label="指示"),
        Textbox(lines=5, label="入力"),
        gr.Slider(
            label='Temperature',
            minimum=0.1,
            maximum=1.0,
            step=0.1,
            value=0.7,
        )
    ],
    Textbox(label="japanese-stablelm-base-gamma-7b"),
)

demo4 = gr.Interface(
    fn4,
    [
        Textbox(lines=5, label="指示"),
        Textbox(lines=5, label="入力"),
        gr.Slider(
            label='Temperature',
            minimum=0.1,
            maximum=1.0,
            step=0.1,
            value=0.7,
        )
    ],
    Textbox(label="japanese-stablelm-instruct-gamma-7b"),
)

demo = gr.Parallel(
    demo_echo,
    demo1,
    demo2,
    demo3,
    demo4,
    examples=examples,
    title="StabilityAIの日本語LLM比較用デモ",
    description="""
        以下の4つのモデルを同時に実行します。
        ・30億パラメータの汎用言語モデル: Japanese Stable LM 3B-4E1T Base
        ・30億パラメータの指示応答言語モデル: Japanese Stable LM 3B-4E1T Instruct
        ・70億パラメータの汎用言語モデル Japanese Stable LM Base Gamma 7B
        ・70億パラメータの指示応答言語モデル Japanese Stable LM Instruct Gamma 7B

        """,)
demo.launch(server_name="0.0.0.0", server_port=8001, share=False)