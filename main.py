import os

import json
from src.llm_handling import HuggingChatHandler
from src.retrieval_handling import VaultIndex
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI as fOpenAI

config = json.load(open("config.json", "r"))

query = input("which question may I help you with?")
vault_index = VaultIndex(embedder_checkpoint=config["EMBEDDER_CHECKPOINT"])
chat_handler = HuggingChatHandler(huggingface_mail=config["HUGGINGFACE_MAIL"], huggingface_pw=config["HUGGINGFACE_PW"])

if not os.path.exists(config["VAULT_DIR"]):
    vault_index.build_index(index_df_path=config["INDEX_PATH"], vault_dir=config["VAULT_DIR"])
# todo split query into multiple parts and do one retrieval per part
top_contents, _ = vault_index.topk(query=query, index_df_path=config["INDEX_PATH"], k=config["K"])
print(top_contents)
print(chat_handler.respond(query=query, contents=top_contents))

