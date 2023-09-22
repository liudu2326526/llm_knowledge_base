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


def multiply_partition(index, iterator):
  model = Bert4Vec(mode=MODEL_TYPE)
  num = 0
  for id, title, content, content_clean, brand_bame_list in iterator:
    try:
      process_content = f'{brand_bame_list}品牌营销文案，标题：{title}。内容：{content_clean}'
      vec = model.encode(process_content,
                         batch_size=DIMENSION, convert_to_numpy=True,
                         normalize_to_unit=False)
      num = num + 1
      logging.info(f'{index} 分区完成向量化数量: {num}')
      yield (id, title, content, content_clean, brand_bame_list,
             vec.tolist())
    except Exception as e:
      logging.error(f'数据写入失败: {e}')
      logging.error(f'{id}:{content_clean}')


# 创建 SparkSession
spark = SparkSession.builder.appName("SparkSQLDemo").getOrCreate()

# df = spark.createDataFrame(data, columns)
df = spark.sql(
    """
select
id,title,content,content_clean,brand_bame_list
from
  (
    select
      id,title,content,content_clean,brand_bame_list,
      row_number() over(
        partition by md5(content_clean)
        order by
          id
      ) rn
    from
      prod_dwd.dwd_pangu_conv_assessment_dataset_ss
  )
where
  rn = 1;
    """)

rdd = df.rdd.repartition(6)

schema = df.schema

# 使用 mapPartitions 进行转换操作
result_rdd = rdd.mapPartitionsWithIndex(multiply_partition)

result_df = spark.createDataFrame(result_rdd, schema.add(
    StructField("embedding", ArrayType(DoubleType()), True)))

result_df.createOrReplaceTempView("embedding_table")

spark.sql("""
INSERT OVERWRITE TABLE prod_mlm.dws_pangu_conv_assessment_embedding_dataset_ss  
SELECT
id,title,content,content_clean,brand_bame_list,embedding
FROM embedding_table
""")

# 关闭 SparkSession
spark.stop()
