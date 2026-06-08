from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", encode_kwargs={"normalize_embeddings": True})

