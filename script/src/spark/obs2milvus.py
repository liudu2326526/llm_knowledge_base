from bert4vec import Bert4Vec
from pyspark.sql import SparkSession
import logging

DIMENSION = 768  # Embeddings size
COUNT = 100  # How many titles to embed and insert.
MILVUS_HOST = '172.16.5.106'  # Milvus server URI
MILVUS_PORT = '19530'
MODEL_TYPE = 'roformer-sim-base'
# MODEL_TYPE = 'simbert-base'
COLLECTION_NAME = 'test_db_' + MODEL_TYPE.replace('-', '_')

# 创建 SparkSession
spark = SparkSession.builder.appName("SparkSQLDemo").getOrCreate()

# 创建一个 DataFrame
data = [("Alice", "teststset"), ("Bob", "dsdfsfsdfd"), ("Charlie", "dsfdsfsf"),
        ("David", "gfsgrsg")]
columns = ["Name", "content"]

# df = spark.createDataFrame(data, columns)
df = spark.sql(
    """
  SELECT
    a.dwd_content_id AS name,
    a.content, 
    a.brand_name,
    a.category_name,
    b.name source_name
  from
    (SELECT
    dwd_content_id,
    MAX(content) content ,
    MAX(source) source,
    concat_ws(',', collect_set(brand_name)) brand_name,
    concat_ws(',', collect_set(category_name)) category_name
  from
    prod_dws.dws_main_content_wide_dt_ctime_daily_inc_30d
  where
    dt = '2023-08-13'
    and length(content) > 5
    and interact_cnt > 0
    and category_name = '纯牛奶'
  group by
    dwd_content_id) a
    JOIN prod_dim.dim_source b on a.source = b.source
  limit
    1000
    """)

rdd = df.rdd.repartition(6)


def multiply_partition(iterator):
  model = Bert4Vec(mode=MODEL_TYPE)
  from pymilvus import connections, Collection
  connections.connect("default", host=MILVUS_HOST, port="19530")
  collection = Collection(name=COLLECTION_NAME)
  collection.load()
  for name, content, brand_name, category_name, source_name in iterator:
    try:
      vec = model.encode(content + brand_name + category_name + source_name,
                         batch_size=DIMENSION, convert_to_numpy=True,
                         normalize_to_unit=False)
      content_sub = str((content[:199]) if len(content) > 200 else content)
      ins = [[name], [content_sub], [vec]]
      collection.insert(ins)
    except Exception as e:
      logging.error(f'数据写入失败: {e}')
      logging.error(f'{name}:{content}')


# 使用 mapPartitions 进行转换操作
result_rdd = rdd.foreachPartition(multiply_partition)

# 关闭 SparkSession
spark.stop()
