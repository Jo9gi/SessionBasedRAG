import os
from langchain_milvus import Milvus
from .embedding import embedding_model


# =====================================================
# CREATE VECTOR DATABASE
# =====================================================

def create_vector_store(chunks):

    vector_store = Milvus.from_documents(

        documents=chunks,

        embedding=embedding_model,

        connection_args={

            "uri": os.environ.get("MILVUS_URI", "http://localhost:19530")
        },

        collection_name="rag_collection",

        drop_old=True,

        metadata_field="metadata",

        index_params={

            "index_type": "HNSW",

            "metric_type": "COSINE",

            "params": {

                "M": 16,

                "efConstruction": 200
            }
        }
    )

    return vector_store