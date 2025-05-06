import json

with open('output/tc_200_zh_output_agent_1.json', 'r', encoding='utf-8') as file:
    data = json.load(file)  # 解析为 Python 字典/列表

# 使用数据
# print(data['results'])

acc = 0
req = 0
opt = 0

for i in range(200):
    acc += data['results'][i]['accuracy']
    req += data['results'][i]['req_acc']
    opt += data['results'][i]['opt_acc']

    # print(data['results'][i]['accuracy'])
    # print(data['results'][i]['req_acc'])
    # print(data['results'][i]['opt_acc'])

print(acc/200, req/200, opt/200)