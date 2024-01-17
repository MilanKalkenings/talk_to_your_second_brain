from typing import List, Tuple
import os
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


class VaultIndex:
    def __init__(self, embedder_checkpoint: str):
        self.tokenizer = AutoTokenizer.from_pretrained(embedder_checkpoint)
        self.model = AutoModel.from_pretrained(embedder_checkpoint)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        with torch.no_grad():
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9)

    @staticmethod
    def read_file(path: str) -> str:
        with open(path, "r", encoding="utf8") as file:
            text = file.read()
        return text

    @staticmethod
    def list_markdowns(dir: str) -> List[str]:
        markdown_files = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(".md"):
                    markdown_files.append(os.path.join(root, file))
        return markdown_files

    def build_index(self, index_df_path: str, vault_dir: str):
        markdown_paths = self.list_markdowns(dir=vault_dir)
        markdown_contents = [self.read_file(path) for path in markdown_paths]
        markdown_contents = [markdown_paths[i] + "\n\n" + markdown_contents[i] for i in range(len(markdown_paths))]
        # todo: so far only docs until certain length are "valid"
        embeddings_valid = []
        markdown_paths_valid = []
        markdown_contents_valid = []
        for i in range(len(markdown_paths)):
            try:
                embeddings_valid.append(self.embed(markdown_contents[i]))
                markdown_paths_valid.append(markdown_paths[i])
                markdown_contents_valid.append(markdown_contents[i])
            except:
                pass
        pd.DataFrame({"path": markdown_paths_valid,
                      "embedding": embeddings_valid,
                      "content": markdown_contents_valid}).to_pickle(index_df_path)

    def embed(self, string: str):
        with torch.no_grad():
            encoded_input = self.tokenizer([string], padding=True, truncation=True, return_tensors='pt')
            model_output = self.model(**encoded_input)
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            return torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

    def topk(self, query: str, index_df_path: str, k: int) -> Tuple[List[str], List[str]]:
        with torch.no_grad():
            query_embedding = self.embed(string=query)[0]
            index_df = pd.read_pickle(index_df_path)
            embeddings = torch.cat([e for e in index_df["embedding"].values])
            distances = [float(torch.norm(query_embedding - doc_embedding, dim=0)) for doc_embedding in embeddings]
            topk_indices = [e.item() for e in torch.argsort(torch.tensor(distances), descending=False)[:k]]
            topk_paths = [index_df["path"][i] for i in topk_indices]
            topk_contents = [index_df["content"][i] for i in topk_indices]
            return topk_contents, topk_paths


