import os

os.environ["OPENAI_API_KEY"] = "<open-ai-key>"

from haystack.components.builders import PromptBuilder

from haystack import Pipeline, Document
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder


# prompt_template = """
# Generate a list of 3 titles for my autobiography. The book is about my journey as an adventurer who has lived an unconventional life, meeting many different personalities and finally finding peace in gardening. Each title should be between two and five words long.
#
# ### Examples of great titles ###
#
# - “Long walk to freedom”
# - “Wishful drinking”
# - “I know why the caged bird sings”
# """


# prompt_template = """
#                 Classify the text into positive, negative, neutral
#                 Text: I do not hate horror movies
#             """


prompt_template = """
                Classify the text into positive, negative, neutral
                Text: The final episode was surprising with a terrible twist at the end
                """


p = Pipeline()
p.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
p.add_component(instance=OpenAIGenerator(), name="llm")
p.connect("prompt_builder", "llm")

result = p.run({"prompt_builder": {}})
print(result)