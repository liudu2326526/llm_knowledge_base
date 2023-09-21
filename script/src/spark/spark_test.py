from pyspark.sql import SparkSession
from pyspark.sql.types import *

# 创建一个SparkSession
spark = SparkSession.builder.appName("RDD to DataFrame").getOrCreate()
rdd = spark.sparkContext.parallelize([(1, "Alice"), (2, "Bob"), (3, "Charlie")])

schema = StructType([
  StructField("id", IntegerType(), True),
  StructField("name", StringType(), True)
])
df = spark.createDataFrame(rdd, schema)
df.createOrReplaceTempView("people")  # 创建一个临时视图

rdd = df.rdd


def multiply_partition(iterator):
  result = []
  for i in iterator:
    print(i)
    result.append(i)
  return result


result_rdd = rdd.mapPartitions(multiply_partition)

df = spark.createDataFrame(rdd, schema)

df.show()

# print(df.schema.add(StructField("embedding", ArrayType(DoubleType()), True)))

# 执行SQL查询
# result = spark.sql("SELECT * FROM people WHERE id > 1")
# result.show()
