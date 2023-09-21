from pymilvus import (
  connections,
  utility,
  FieldSchema,
  CollectionSchema,
  DataType,
  Collection
)

MODEL_TYPE = 'roformer-sim-base'
# MODEL_TYPE = 'simbert-base'
COLLECTION_NAME = 'test_db_' + MODEL_TYPE.replace('-', '_')

connections.connect("default", host="172.16.5.106", port="19530")

schema = CollectionSchema(
    fields=[
      FieldSchema(name='id', dtype=DataType.VARCHAR, descrition='Ids',
                  is_primary=True, auto_id=False, max_length=200),
      FieldSchema(name='title', dtype=DataType.VARCHAR,
                  description='Title texts',
                  max_length=200),
      FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR,
                  description='Embedding vectors', dim=768)
    ],
    description="Test book search",
    enable_dynamic_field=True
)
collection_name = COLLECTION_NAME

collection = Collection(
    name=collection_name,
    schema=schema,
    using='default',
    shards_num=2
)
