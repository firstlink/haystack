import os

os.environ["OPENAI_API_KEY"] = "<open-ai-key>"

from haystack import Pipeline
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder


prompt_template = """
I went to the market and bought 10 apples. 
I gave 2 apples to the neighbor and 2 to the repairman. 
I then went and bought 5 more apples and ate 1. How many apples did I remain with?
Take step by step approach
"""

p = Pipeline()
p.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
p.add_component(instance=OpenAIGenerator(), name="llm")
p.connect("prompt_builder", "llm")

result = p.run({"prompt_builder": {}})
print(result)