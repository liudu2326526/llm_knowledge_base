from flask import Flask, request, jsonify
from flask_cors import CORS
from milvus_db import bert_search

app = Flask(__name__)
CORS(app)  # 允许所有域名的请求

# 定义一个路由，接受查询参数，并返回数据
@app.route('/search', methods=['GET'])
def search():
  # 获取查询参数中的 'query' 参数
  query = request.args.get('query')

  # 这里可以编写处理查询参数并返回数据的逻辑
  # 示例：简单返回一个字典作为响应数据
  response_data = {
    'query': query,
    'results': bert_search.search(query)
  }
  print(response_data)
  # 将字典转换为 JSON 格式，并返回
  return jsonify(response_data)

if __name__ == '__main__':
  app.run(debug=True,port=5001)