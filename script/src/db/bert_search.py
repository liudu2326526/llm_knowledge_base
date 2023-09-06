from arithmetic import bert_doc2vec
from pymilvus import connections, Collection

MILVUS_HOST = '172.16.5.106'  # Milvus server URI
MILVUS_PORT = '19530'
MODEL_TYPE = 'roformer-sim-base'
# MODEL_TYPE = 'simbert-base'
COLLECTION_NAME = 'test_db_' + MODEL_TYPE.replace('-', '_')


connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
collection = Collection(name=COLLECTION_NAME)
collection.load()

def search(text):
  # Search parameters for the index
  search_params = {
    "metric_type": "L2"
  }

  results = collection.search(
      data=[bert_doc2vec.embed(text)],  # Embeded search value
      anns_field="embedding",  # Search across embeddings
      param=search_params,
      limit=10,  # Limit to five results per search
      output_fields=['title']  # Include title field in result
  )

  ret = []
  for hit in results[0]:
    row = []
    row.extend([hit.id, hit.score, hit.entity.get(
        'title')])  # Get the id, distance, and title for the results
    ret.append(row)
  return ret
