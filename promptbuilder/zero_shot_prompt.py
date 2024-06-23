import os
from haystack import Pipeline, Document
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder

os.environ["OPENAI_API_KEY"] = "<open-ai-key>"

# prompt_template = """
#                   Write a short paragraph with 100 words about Haystack
#                   """

prompt_template = """
                  Write a short paragraph with 100 words about Haystack, an open-source nlp framework.
                  """

p = Pipeline()
p.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
p.add_component(instance=OpenAIGenerator(), name="llm")
p.connect("prompt_builder", "llm")

result = p.run({"prompt_builder": {}})
print(result)