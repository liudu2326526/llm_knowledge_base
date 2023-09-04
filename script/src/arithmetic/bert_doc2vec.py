import csv
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
    print(embed("数据测试"))