from haystack import Document
from haystack.components.retrievers import FilterRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

docs = [
    Document(content="Python is a popular programming language", meta={"lang": "en"}),
    Document(content="python ist eine beliebte Programmiersprache", meta={"lang": "de"}),
]

doc_store = InMemoryDocumentStore()
doc_store.write_documents(docs)

retriever = FilterRetriever(doc_store)
results = retriever.run(filters={"field": "lang", "operator": "==", "value": "de"})

for result in results["documents"]:
    print(result.content)

