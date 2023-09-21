from bert4vec import Bert4Vec
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import logging

DIMENSION = 768  # Embeddings size
MODEL_TYPE = 'roformer-sim-base'

source_map = {
  1013: '微博',
  1014: '微信',
  1027: '小红书',
  1026: '快手',
  1012: '抖音',
  1008: '知乎',
  1022: 'B站',
}


def multiply_partition(iterator):
  model = Bert4Vec(mode=MODEL_TYPE)
  for dwd_content_id, source, text, content_clean, brandName, className in iterator:
    try:
      process_content = f'{source_map[source]}平台的文案，来自{className}行业的{brandName}品牌营销内容：{content_clean}'
      vec = model.encode(process_content,
                         batch_size=DIMENSION, convert_to_numpy=True,
                         normalize_to_unit=False)
      yield (dwd_content_id, source, text, content_clean, brandName, className,
             vec.tolist())
    except Exception as e:
      logging.error(f'数据写入失败: {e}')
      logging.error(f'{dwd_content_id}:{content_clean}')


# 创建 SparkSession
spark = SparkSession.builder.appName("SparkSQLDemo").getOrCreate()

# df = spark.createDataFrame(data, columns)
df = spark.sql(
    """
SELECT
  dwd_content_id,
  source,
  text,
  content_clean,
  brandName,
  className
FROM
  prod_mlm.dws_huawei_marketing_tag_ss
where
  label = '0'
  and score > 0.95
limit 1000
    """)

rdd = df.rdd.repartition(6)

schema = df.schema

# 使用 mapPartitions 进行转换操作
result_rdd = rdd.mapPartitions(multiply_partition)

result_df = spark.createDataFrame(result_rdd, schema.add(
    StructField("embedding", ArrayType(DoubleType()), True)))

result_df.createOrReplaceTempView("embedding_table")

spark.sql("""
INSERT OVERWRITE TABLE prod_mlm.dws_huawei_embedding_tag_ss  
SELECT
  dwd_content_id,
  source,
  text,
  content_clean,
  brandName,
  className,
  embedding
FROM embedding_table
""")

# 关闭 SparkSession
spark.stop()
