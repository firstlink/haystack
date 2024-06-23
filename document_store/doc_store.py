from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import Document

documents = [
    Document(id=1, content="This is document 1"),
    Document(id=1, content="This is document 2")
]

document_store = InMemoryDocumentStore()
embedder = SentenceTransformersDocumentEmbedder()

document_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.NONE)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component(name="embedder", instance=embedder)
indexing_pipeline.add_component(name="writer", instance=document_writer)
indexing_pipeline.connect("embedder", "writer")
results = indexing_pipeline.run({"embedder":{"documents": documents}})
print(results)
