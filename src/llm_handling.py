import os  # file handling
from typing import List  # type hints for improved readability
import time  # to wait some time between api calls
import numpy as np  # array handling
from numpy.typing import NDArray  # type hints for improved readability
from numpy import float64  # type hints for improved readability
import pandas as pd  # data loading
from sklearn.model_selection import train_test_split  # to evaluate on data on which classifier has not been trained
import torch  # embedding handling
from transformers import AutoModel, AutoTokenizer  # embedding handling
from hugchat import hugchat  # chat with hugging face LLMs
from hugchat.login import Login  # hugging face auth
from sklearn.metrics import accuracy_score, precision_score, recall_score  # classification evaluation


class HuggingChatHandler:
    def __init__(self, huggingface_mail: str, huggingface_pw: str):
        sign = Login(huggingface_mail, huggingface_pw)
        cookies = sign.login()
        cookie_path_dir = "../../monitoring/cookies_snapshot"
        sign.saveCookiesToDir(cookie_path_dir)
        self.llm = hugchat.ChatBot(cookies=cookies.get_dict())

    def respond(self, query: str, contents: List[str]) -> str:
        # todo: get reference from prompt to refer back to useful files
        prompt = "You are an expert in machine learning. Given are some flashcards:"
        prompt += "\n\n".join([f"Document {i + 1}:\n'''" + contents[i] + "'''" for i in range(len(contents))])
        prompt += f"\n\nAnswer the question '''{query}'''! Some of your flashcards might help you. Respond in bullet points and keep it short while covering important information"
        while True:
            try:
                response = str(self.llm.query(prompt))
                break
            except Exception as e:
                print(e)
                time.sleep(600)
        return response

    def respond_baseline(self, query: str):
        prompt = "You are an expert in machine learning."
        prompt += f"\n\nAnswer the question '''{query}'''! Respond in bullet points and keep it short while covering important information"
        while True:
            try:
                response = str(self.llm.query(prompt))
                break
            except Exception as e:
                print(e)
                time.sleep(600)
        return response


class Parser:
    """
    used to parse binary prediction from LLM-output string
    """
    @staticmethod
    def extract_substring(text: str, start_sub: str, end_sub: str) -> str:
        start_idx = text.find(start_sub)
        if start_idx == -1:
            return ""  # Start substring not found
        start_idx += len(start_sub)
        end_idx = text.find(end_sub, start_idx)
        if end_idx == -1:
            return ""  # End substring not found
        return text[start_idx:end_idx]