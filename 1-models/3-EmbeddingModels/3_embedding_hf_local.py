from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "New Delhi is the capital of India",
    "Lucknow is the capital of Uttar Pradesh",
    "Paris is the capital of France"
]

vector = embedding.embed_documents(documents)

print(str(vector))
