import os.path

from src.llm_handling import HuggingChatHandler
from config import HUGGINGFACE_MAIL, HUGGINGFACE_PW, VAULT_DIR, INDEX_PATH, EMBEDDER_CHECKPOINT, K
from src.retrieval_handling import VaultIndex

query = input("which question may I help you with?")
vault_index = VaultIndex(embedder_checkpoint=EMBEDDER_CHECKPOINT)
chat_handler = HuggingChatHandler(huggingface_mail=HUGGINGFACE_MAIL, huggingface_pw=HUGGINGFACE_PW)

if not os.path.exists(VAULT_DIR):
    vault_index.build_index(index_df_path=INDEX_PATH, vault_dir=VAULT_DIR)
# todo split query into multiple parts and do one retrieval per part
top_contents, _ = vault_index.topk(query=query, index_df_path=INDEX_PATH, k=K)
print(top_contents)
print(chat_handler.respond(query=query, contents=top_contents))

