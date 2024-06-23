from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
text_embedder = SentenceTransformersTextEmbedder()
retriever = InMemoryEmbeddingRetriever(document_store=document_store)

query_pipline = Pipeline()

query_pipline.add_component("text_embedder", text_embedder)
query_pipline.add_component("retriever", retriever)

query_pipline.connect("text_embedder.embedding", "retriever.query_embedding")
query_pipline.connect("text_embedder.embedding", "retriever")


query = query = "Here comes the query text"
query_pipline.run({"text_embedder": {"text": query}})


