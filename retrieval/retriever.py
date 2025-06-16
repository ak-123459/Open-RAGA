from retrieval.vector_store import get_vector_store


# Function to retrieve similar type vectors
def similar_retriever(query,topk=1):

  vectors = get_vector_store()

  results = vectors.similarity_search(query, topk)

  context = results
  
  return context


