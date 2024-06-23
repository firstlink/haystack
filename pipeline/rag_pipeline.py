import os

from datasets import load_dataset
from haystack import Document
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack import Pipeline


dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
docs = []
for doc in dataset:
    # print(f"Doc from HF - {doc} \n")
    docs.append(Document(content=doc["content"], meta=doc["meta"]))

document_store = InMemoryDocumentStore()

doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()
docs_with_embeddings = doc_embedder.run(docs)

for doc in docs_with_embeddings["documents"]:
    print(f"Doc with embedding -{doc.embedding}")

document_store.write_documents(docs_with_embeddings["documents"])
retriever = InMemoryEmbeddingRetriever(document_store)

text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
text_embedder.warm_up()


template = """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""

prompt_builder = PromptBuilder(template=template)

os.environ["OPENAI_API_KEY"] = "<open-ai-key>"
generator = OpenAIGenerator(model="gpt-3.5-turbo")

basic_rag_pipeline = Pipeline()

# Add components to your pipeline
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", generator)


basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
basic_rag_pipeline.connect("prompt_builder", "llm")

question = "What does Rhodes Statue look like?"
response = basic_rag_pipeline.run({"text_embedder": {"text": question}, "prompt_builder": {"question": question}})
print("Response from Generator : ", response["llm"]["replies"][0])