from haystack import Document
from haystack.components.joiners.document_joiner import DocumentJoiner

docs_1 = [Document(content="Paris is the capital of France.", score=0.5), Document(content="Berlin is the capital of Germany.", score=0.4)]
docs_2 = [Document(content="Paris is the capital of France.", score=0.6), Document(content="Rome is the capital of Italy.", score=0.5)]

joiner = DocumentJoiner(join_mode="reciprocal_rank_fusion")

result = joiner.run(documents=[docs_1, docs_2])
print(result)