nice_animals=["Capybara", "Dolphin"]


import wikipedia
from haystack.dataclasses import Document

raw_docs = []
for title in nice_animals:
    page = wikipedia.page(title=title, auto_suggest=False)
    doc = Document(content=page.content, meta={"title": page.title, "url":page.url})
    raw_docs.append(doc)


from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

document_store = QdrantDocumentStore(
    ":memory:",
    recreate_index=True,
    return_embedding=True,
    use_sparse_embeddings=True  # set this parameter to True, otherwise the collection schema won't allow to store sparse vectors
)


from haystack_integrations.components.embedders.fastembed import FastembedSparseDocumentEmbedder

sparse_doc_embedder = FastembedSparseDocumentEmbedder(model="prithvida/Splade_PP_en_v1",
                                                      meta_fields_to_embed=["title"])
sparse_doc_embedder.warm_up()

# let's try the embedder
print(sparse_doc_embedder.run(documents=[Document(content="An example document")]))


from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack import Pipeline

indexing = Pipeline()
indexing.add_component("cleaner", DocumentCleaner())
indexing.add_component("splitter", DocumentSplitter(split_by='sentence', split_length=4))
indexing.add_component("sparse_doc_embedder", sparse_doc_embedder)
indexing.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE))

indexing.connect("cleaner", "splitter")
indexing.connect("splitter", "sparse_doc_embedder")
indexing.connect("sparse_doc_embedder", "writer")

indexing.run({"documents":raw_docs})


from haystack import Pipeline
from haystack_integrations.components.retrievers.qdrant import QdrantSparseEmbeddingRetriever
from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder

sparse_text_embedder = FastembedSparseTextEmbedder(model="prithvida/Splade_PP_en_v1")

query_pipeline = Pipeline()
query_pipeline.add_component("sparse_text_embedder", sparse_text_embedder)
query_pipeline.add_component("sparse_retriever", QdrantSparseEmbeddingRetriever(document_store=document_store))

query_pipeline.connect("sparse_text_embedder.sparse_embedding", "sparse_retriever.query_sparse_embedding")

question = "Where do capybaras live?"
results = query_pipeline.run({"sparse_text_embedder": {"text": question}})

for d in results['sparse_retriever']['documents']:
  print(f"\nid: {d.id}\n{d.content}\nscore: {d.score}\n---")