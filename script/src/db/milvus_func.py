from pymilvus import (
  connections,
  utility,
  FieldSchema,
  CollectionSchema,
  DataType,
  Collection
)


DIMENSION = 768  # Embeddings size
COUNT = 100  # How many titles to embed and insert.
MILVUS_HOST = '172.16.5.106'  # Milvus server URI
MILVUS_PORT = '19530'
MODEL_TYPE = 'roformer-sim-base'
# MODEL_TYPE = 'simbert-base'
COLLECTION_NAME = 'pangu_data_' + MODEL_TYPE.replace('-', '_')

connections.connect("default", host="172.16.5.106", port="19530")


def create():
  fields = [
    FieldSchema(name='id', dtype=DataType.VARCHAR, descrition='Ids',
                is_primary=True, auto_id=False, max_length=200),
    FieldSchema(name='title', dtype=DataType.VARCHAR, description='Title texts',
                max_length=200),
    FieldSchema(name='content', dtype=DataType.VARCHAR, description='Content texts',
                max_length=500),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR,
                description='Embedding vectors', dim=DIMENSION)
  ]
  schema = CollectionSchema(fields=fields, description='Title collection')
  collection = Collection(name=COLLECTION_NAME, schema=schema)

  index_params = {
    'index_type': 'IVF_FLAT',
    'metric_type': 'L2',
    'params': {'nlist': 1024}
  }
  collection.create_index(field_name="embedding", index_params=index_params)


create()
