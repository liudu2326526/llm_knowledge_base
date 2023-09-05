from bert4vec import Bert4Vec

DIMENSION = 768  # Embeddings size
MODEL_TYPE = 'roformer-sim-base'

model = Bert4Vec(mode=MODEL_TYPE)

# Extract embedding from text using OpenAI
def embed(text):
  print(text)
  return model.encode(text, batch_size=DIMENSION, convert_to_numpy=True,
                      normalize_to_unit=False)




if __name__ == '__main__':
    print(embed("""
【Calzedonia和Intimissimi发布全新2023秋冬系列】
#时尚生活实验室# 意大利裤袜品牌Calzedonia与内衣家居服品牌Intimissimi以The Art of Italian Fashion为主题于上海东一美术馆发布全新2023秋冬系列，呈现意大利艺术与时尚的融合。
#裤袜新品##内衣新品##2023秋冬系列##Calzedonia##Intimissimi#
    """))