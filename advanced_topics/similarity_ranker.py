from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.rankers import TransformersSimilarityRanker

docs = [Document(content="Paris is in France"),
        Document(content="Berlin is in Germany"),
        Document(content="Lyon is in France")]

document_store = InMemoryDocumentStore()
document_store.write_documents(docs)

retriever = InMemoryBM25Retriever(document_store=document_store)
ranker = TransformersSimilarityRanker()
ranker.warm_up()

document_ranker_pipeline = Pipeline()
document_ranker_pipeline.add_component(instance=retriever, name="retriever")
document_ranker_pipeline.add_component(instance=ranker, name="ranker")

document_ranker_pipeline.connect("retriever.documents", "ranker.documents")

query = "Cities in France"
result = document_ranker_pipeline.run(data={"retriever": {"query": query, "top_k": 3},
                                   "ranker": {"query": query, "top_k": 2}})

for d in result["ranker"]["documents"]:
    print(d.content, d.score)
