from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents = [
    "Black holes are regions of spacetime where gravity is so strong that not even light can escape.",
    "The Big Bang theory describes the universe expanding from a hot, dense state around 13.8 billion years ago.",
    "Dark matter is an invisible form of matter that does not emit light but shapes the structure of galaxies.",
    "Exoplanets are planets that orbit stars outside our solar system.",
    "The James Webb Space Telescope uses infrared to reveal some of the earliest galaxies ever seen."
]

query = 'tell me about dark matter'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)
