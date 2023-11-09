from bert4vec import Bert4Vec
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import logging


# 创建 SparkSession
spark = SparkSession.builder.appName("SparkSQLDemo").getOrCreate()

# df = spark.createDataFrame(data, columns)
df = spark.sql(
    """
INSERT OVERWRITE TABLE prod_mlm.know_base_pangu_baichuan_sft_rqa_train_daily_ss 
select 
  dwd_content_id,source,title,content,share_url,brand_3class,baichuan2_keyword,baichuan2_keyword_clean,
  filter(split(replace(replace(brand_name_list,']',''),'[',''),','),x->x IS NOT NULL OR length(x) > 0) brand_name_list,
  filter(split(replace(replace(category_name_list,']',''),'[',''),','),x->x IS NOT NULL OR length(x) > 0) category_name_list,
  filter(split(replace(replace(brand_keyword,']',''),'[',''),','),x->x IS NOT NULL OR length(x) > 0) brand_keyword,
  filter(split(replace(replace(scenes_keyword_list,']',''),'[',''),','),x->x IS NOT NULL OR length(x) > 0) scenes_keyword_list,
  from_json(content_style, 'ARRAY<STRING>') content_style,
  filter(split(replace(replace(crowd_keyword_list,']',''),'[',''),','),x->x IS NOT NULL OR length(x) > 0) crowd_keyword_list,
  filter(split(replace(replace(dim_keyword_level3_list,']',''),'[',''),','),x->x IS NOT NULL OR length(x) > 0) dim_keyword_level3_list,
  filter(split(replace(replace(dim_keyword_level4_list,']',''),'[',''),','),x->x IS NOT NULL OR length(x) > 0) dim_keyword_level4_list
from prod_mlm.fdl_pangu_baichuan_sft_rqa_train_parq_daily_ss
    """)

# 关闭 SparkSession
spark.stop()
