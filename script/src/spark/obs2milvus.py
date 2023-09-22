from bert4vec import Bert4Vec
from pyspark.sql import SparkSession
import logging

DIMENSION = 768  # Embeddings size
COUNT = 100  # How many titles to embed and insert.
MILVUS_HOST = '172.16.5.106'  # Milvus server URI
MILVUS_PORT = '19530'
MODEL_TYPE = 'roformer-sim-base'
# MODEL_TYPE = 'simbert-base'
COLLECTION_NAME = 'pangu_data_' + MODEL_TYPE.replace('-', '_')

# 创建 SparkSession
spark = SparkSession.builder.appName("SparkSQLDemo").getOrCreate()

df = spark.sql(
    """
select
  id,
  title,
  content_clean,
  embedding
from
  prod_mlm.dws_pangu_conv_assessment_embedding_dataset_ss 
    """)

rdd = df.rdd.repartition(6)


def multiply_partition(iterator):
  from pymilvus import connections, Collection
  connections.connect("default", host=MILVUS_HOST, port="19530")
  collection = Collection(name=COLLECTION_NAME)
  collection.load()
  for id, title, content_clean, embedding in iterator:
    try:
      content_sub = str(
        (content_clean[:499]) if len(content_clean) > 500 else content_clean)
      title_sub = str((title[:199]) if len(title) > 200 else title)
      ins = [[id], [title_sub], [content_sub], [embedding]]
      collection.insert(ins)
    except Exception as e:
      logging.error(f'数据写入失败: {e}')
      logging.error(f'{id}:{content_clean}')


# 使用 mapPartitions 进行转换操作
result_rdd = rdd.foreachPartition(multiply_partition)

# 关闭 SparkSession
spark.stop()
