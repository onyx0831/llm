prompt_template_dict = {
    'alpaca': (
        "以下は、タスクを説明する指示です。"
        "要求を適切に満たす応答を書きなさい。\n\n"
        "### 指示:\n{instruction}\n\n### 応答:"
    ),
    'line_llm': (
        "ユーザー: {instruction}\nシステム: "
    )
}