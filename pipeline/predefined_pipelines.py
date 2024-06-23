import urllib.request
from haystack import Pipeline, PredefinedPipeline
import os

urllib.request.urlretrieve("https://www.gutenberg.org/cache/epub/7785/pg7785.txt", "../davinci.txt")

os.environ["OPENAI_API_KEY"] = "<open-ai-key>"

# Indexing Pipeline
# Generates a pipeline that imports documents from one or more text files,
# creates the embeddings for each of them and finally stores them into a Chroma database.
indexing_pipeline = Pipeline.from_template(PredefinedPipeline.INDEXING)
indexing_pipeline.draw(path="predefined_index_pipeline.jpg")
indexing_pipeline.run(data={"sources": ["davinci.txt"]})

# RAG Pipeline
# Generates a RAG pipeline using data that was previously indexed (you can use the Indexing template).
rag_pipeline = Pipeline.from_template(PredefinedPipeline.RAG)

rag_pipeline.draw(path="predefined_rag_pipeline.jpg")
query = "How old was he when he died?"
result = rag_pipeline.run(
    data={"prompt_builder": {"query": query}, "text_embedder": {"text": query}}
)
print(result["llm"]["replies"][0])