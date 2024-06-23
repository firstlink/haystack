import os
from haystack import Pipeline, Document
from haystack.utils import Secret
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder

os.environ["OPENAI_API_KEY"] = "<open-ai-key>"

# in a real world use case documents could come from a retriever, web, or any other source
documents = [Document(content="Joe lives in Berlin"), Document(content="Joe is a software engineer")]
prompt_template = """
    Given these documents, answer the question.\nDocuments:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    \nQuestion: {{query}}
    \nAnswer:
    """
p = Pipeline()
p.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
p.add_component(instance=OpenAIGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY")), name="llm")
p.connect("prompt_builder", "llm")

question = "Where does Joe live?"
result = p.run({"prompt_builder": {"documents": documents, "query": question}})
print(result)