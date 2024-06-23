from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.fastembed import (
    FastembedTextEmbedder,
    FastembedDocumentEmbedder,
    FastembedSparseTextEmbedder,
    FastembedSparseDocumentEmbedder
)

document_store = QdrantDocumentStore(
    ":memory:",
    recreate_index=True,
    use_sparse_embeddings=True,
    embedding_dim=384
)

documents = [
    Document(content="My name is Wolfgang and I live in Berlin"),
    Document(content="I saw a black horse running"),
    Document(content="Germany has many big cities"),
    Document(content="fastembed is supported by and maintained by Qdrant."),
]

indexing = Pipeline()
indexing.add_component("sparse_doc_embedder", FastembedSparseDocumentEmbedder(model="prithvida/Splade_PP_en_v1"))
indexing.add_component("dense_doc_embedder", FastembedDocumentEmbedder(model="BAAI/bge-small-en-v1.5"))
indexing.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE))
indexing.connect("sparse_doc_embedder", "dense_doc_embedder")
indexing.connect("dense_doc_embedder", "writer")

indexing.run({"sparse_doc_embedder": {"documents": documents}})

querying = Pipeline()
querying.add_component("sparse_text_embedder", FastembedSparseTextEmbedder(model="prithvida/Splade_PP_en_v1"))
querying.add_component("dense_text_embedder", FastembedTextEmbedder(
    model="BAAI/bge-small-en-v1.5", prefix="Represent this sentence for searching relevant passages: ")
                       )
querying.add_component("retriever", QdrantHybridRetriever(document_store=document_store))

querying.connect("sparse_text_embedder.sparse_embedding", "retriever.query_sparse_embedding")
querying.connect("dense_text_embedder.embedding", "retriever.query_embedding")

question = "Who supports fastembed?"

results = querying.run(
    {"dense_text_embedder": {"text": question},
     "sparse_text_embedder": {"text": question}}
)

print(results["retriever"]["documents"][0])