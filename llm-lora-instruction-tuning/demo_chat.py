import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import datetime
import argparse
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = "/media/sj-archimedes/data/03_pretrained_model/llm/elyza/ELYZA-japanese-Llama-2-7b-fast-instruct"


def get_args():
    """
    実行時引数を設定
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="モデルを格納したフォルダのパス",
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=8001,
        help="サーバーポート",
    )
    parser.add_argument(
        "--max_overall_tokens",
        type=int,
        default=1024,
        help="モデルに入力したトークン数がこれを超えたら過去の対話履歴を削除する。",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="１回のチャットの返答の長さの上限",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="温度パラメータ",
    )
    parser.add_argument("--run_on_cpu", action="store_true", help="CPU上で動かす。")
    args = parser.parse_args()
    return args


class ChatModel:
    def __init__(
        self,
        model_path,
        max_new_tokens=512,
        temperature=0.7,
        max_overall_tokens=1024,
        run_on_cpu=False,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_overall_tokens = max_overall_tokens

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.float16, device_map='auto'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        # if torch.cuda.is_available() and not run_on_cpu:
        #     self.model = self.model.to("cuda")
        self.model.eval()
        self.device = self.model.device
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self._INST = "[INST]"
        self.E_INST = "[/INST]"
        self._SYS = "<<SYS>>\n"
        self.E_SYS = "\n<</SYS>>\n\n"

    def _generate_model_input(self, user_input, history):
        """
        新規のインプットと過去の履歴をモデルに入力するための形式に変換する。
        """
        prompt = f"{self.bos_token}{self._INST}"
        for t in history:
            prompt += f"{t[0]} {self.E_INST}"
            prompt += f"{t[1]} {self.eos_token}{self.bos_token}{self._INST}"
        prompt += f"{user_input} {self.E_INST}"
        return prompt

    def _update_history(self, history, user_input, bot_output):
        history.append((user_input, bot_output))
        return history

    def _truncate_history(self, history, truncate_num=1):
        """
        過去の履歴の最初のやりとりを削除する。
        """
        return history[truncate_num:]

    def __call__(self, user_input, history=[]):
        prompt = self._generate_model_input(user_input, history)
        token_ids = self.tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        )

        with torch.no_grad():
            output_ids = self.model.generate(
                token_ids.to(self.device),
                do_sample=True,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        output = self.tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :])
        output = output.replace("<NL>", "\n").rstrip("</s>")

        history = self._update_history(history, user_input, output)

        if token_ids.shape[1] + output_ids.shape[1] > self.max_overall_tokens:
            history = self._truncate_history(history)
        return output, history


def print_log(history):
    print(datetime.datetime.now())
    print(f"history: {history}")


if __name__ == "__main__":
    args = get_args()
    model = ChatModel(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_overall_tokens=args.max_overall_tokens,
        run_on_cpu=args.run_on_cpu,
    )

    with gr.Blocks(title="elyza llm") as demo:
        gr.Markdown(
            """
        # ELYZAのやつ！
        """
        )

        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="入力")
        submit = gr.Button("送信")
        clear = gr.Button("クリア")

        def respond(user_input, chat_history):
            _, chat_history = model(user_input, chat_history)
            print_log(chat_history)
            return "", chat_history

        # msg.submitの行をコメントアウトすると、エンターでは入力確定されない挙動になる。
        msg.submit(fn=respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
        submit.click(fn=respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
        clear.click(fn=lambda: None, inputs=None, outputs=chatbot, queue=False)

    demo.launch(
        share=False, height=2000, server_name="0.0.0.0", server_port=args.server_port
    )
