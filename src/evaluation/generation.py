import os
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI
import json



class Evaluator:
    def __init__(self):
        config = json.load(open("config.json", "r"))
        os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
        self.openai = OpenAI()
        self.trulens_groundedness = Groundedness(groundedness_provider=self.openai, summarize_provider=self.openai)

    def eval(self, source: str, statement: str):
        correctness = self.openai.correctness(statement)
        groundedness = self.trulens_groundedness.groundedness_measure_with_summarize_step(source=source, statement=statement)
        return correctness, groundedness

evaluator = Evaluator()
correctness, groundedness = evaluator.eval(source="cats are fluffy and cute", statement="cats are cute")
print("Correctness:\n", correctness)
print("Groundedness:\n", groundedness)
