import os
from pathlib import Path
from haystack import Pipeline
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.dataclasses import Document
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.document_stores.in_memory import InMemoryDocumentStore


document_store = InMemoryDocumentStore()

# documents = [
#     Document(content="This contains variable declarations", meta={"title": "one"}),
#     Document(content="This contains another sort of variable declarations", meta={"title": "two"}),
#     Document(content="This has nothing to do with variable declarations", meta={"title": "three"}),
#     Document(content="A random doc", meta={"title": "four"}),
# ]

documents = [Document(content="Paris is in France"),
        Document(content="Berlin is in Germany"),
        Document(content="Lyon is in France")]

indexing = Pipeline()
indexing.add_component("embedder", SentenceTransformersDocumentEmbedder())
indexing.add_component("writer", DocumentWriter(document_store))
indexing.connect("embedder.documents", "writer.documents")
indexing.run({"embedder": {"documents": documents}})

querying = Pipeline()
querying.add_component("query_embedding", SentenceTransformersTextEmbedder())
querying.add_component("retriever", InMemoryEmbeddingRetriever(document_store))
querying.connect("query_embedding.embedding", "retriever.query_embedding")

# user_input = "Variable statement"
user_input = "Cities in France"
results = querying.run({"query_embedding": {"text": user_input}})

for d in results["retriever"]["documents"]:
    print(d.content, d.score)

