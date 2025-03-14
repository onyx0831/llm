import copy
from tqdm import tqdm
from torch.utils.data import Dataset


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

class InstructDatasetIgnoreSource(Dataset):
    """
    stanford alpacaの学習コードのように指示文部分をLLMの学習の損失計算に使わないように
    labelsの指示文の箇所をignore indexで埋める形式でdatasetを作成する
    このクラスを使用したときはhuggingfaceのDataCollatorForLanguageModelingは使えない。
    """
    def __init__(self, json_list, tokenizer, ignore_index=-100):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.features = []
        
        for j in tqdm(json_list):
            # open_qaなど文脈情報が必要ない場合はinputカラムがないため、
            # inputカラムありなしでテンプレート文を分けている。
            if 'input' in j:
                source_text = PROMPT_DICT['prompt_input'].format_map(j)
            else:
                source_text = PROMPT_DICT['prompt_no_input'].format_map(j)
            
            # 指示文と回答文を結合し、文末にEOSトークンを挿入
            example_text = source_text + j['output'] + self.tokenizer.eos_token
            
            # 指示文のみ（「以下は、タスクを〜### 応答:」まで）をtokenize
            # ほしいのは指示文のlength
            source_tokenized = self.tokenizer(
                source_text,
                padding='longest',
                truncation=True,
                max_length=512,
                return_length=True,
                return_tensors='pt'
            )
            
            # 指示文と回答文を全てtokenize
            example_tokenized = self.tokenizer(
                example_text, 
                padding='longest', 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            )
            
            input_ids = example_tokenized['input_ids'][0]
            
            # LLMが生成してほしい正解の文章として入力文をそのままコピーする
            labels = copy.deepcopy(input_ids)
            
            # 指示文までの長さ
            source_len = source_tokenized['length'][0]
            
            # LLMに生成してほしい正解文章に指示文も含まれているので、
            # 指示文のところはCrossEntropyLossの損失を計算をしないようにIGNORE_INDEXとして-100で埋める
            labels[:source_len] = self.ignore_index
            
            self.features.append({
                'input_ids': input_ids,
                'labels': labels
            })
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]


class InstructDataset(Dataset):
    """
    指示文もLLMの学習に使用しても良い場合はこちらのクラスを使う。
    こちらのクラスを使うときは、DataCollatorForLanguageModelingが使えるため、
    __getitem__の戻り値にlabelsは含めていない。
    """
    def __init__(self, json_list, tokenizer):
        self.tokenizer = tokenizer
        
        example_texts = []
        for j in json_list:
            # open_qaなど文脈情報が必要ない場合はinputカラムがないため、inputカラムありなしでテンプレート文を分けている。
            if 'input' in j:
                source_text = PROMPT_DICT['prompt_input'].format_map(j)
            else:
                source_text = PROMPT_DICT['prompt_no_input'].format_map(j)
            
            # 指示文と回答文を結合し、文末にEOSトークンを挿入
            example_text = source_text + j['output'] + self.tokenizer.eos_token
            example_texts.append(example_text)
        
        self.features = [
            tokenizer(
                text+self.tokenizer.eos_token, padding=False, truncation=True, max_length=512
            ) for text in tqdm(example_texts)
        ]
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]