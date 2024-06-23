from datetime import datetime

from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever

documents = [
    Document(
        content="Use pip to install a basic version of Haystack's latest release: pip install farm-haystack. All the core Haystack components live in the haystack repo. But there's also the haystack-extras repo which contains components that are not as widely used, and you need to install them separately.",
        meta={"version": 1.15, "date": datetime(2023, 3, 30)},
    ),
    Document(
        content="Use pip to install a basic version of Haystack's latest release: pip install farm-haystack[inference]. All the core Haystack components live in the haystack repo. But there's also the haystack-extras repo which contains components that are not as widely used, and you need to install them separately.",
        meta={"version": 1.22, "date": datetime(2023, 11, 7)},
    ),
    Document(
        content="Use pip to install only the Haystack 2.0 code: pip install haystack-ai. The haystack-ai package is built on the main branch which is an unstable beta version, but it's useful if you want to try the new features as soon as they are merged.",
        meta={"version": 2.0, "date": datetime(2023, 12, 4)},
    ),
]
document_store = InMemoryDocumentStore(bm25_algorithm="BM25Plus")
document_store.write_documents(documents=documents)

pipeline = Pipeline()
pipeline.add_component(instance=InMemoryBM25Retriever(document_store=document_store), name="retriever")

# query = "Haystack installation"
# results = pipeline.run(data={"retriever": {"query": query, "filters": {"field": "meta.version", "operator": ">", "value": 1.22}}})

query = "Haystack installation"
results = pipeline.run(
    data={
        "retriever": {
            "query": query,
            "filters": {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.version", "operator": ">", "value": 1.21},
                    {"field": "meta.date", "operator": ">", "value": datetime(2023, 11, 6)},
                ],
            },
        }
    }
)


for result in results["retriever"]["documents"]:
    print(result)
