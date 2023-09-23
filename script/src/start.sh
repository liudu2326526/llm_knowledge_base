cd ~/llm_knowledge_base/script/src/
nohup python3 -m app.app > ~/llm_knowledge_base/logs/be.log 2>&1 &
nohup python3 -m http.server 9999 > ~/llm_knowledge_base/logs/fe.log 2>&1 &
