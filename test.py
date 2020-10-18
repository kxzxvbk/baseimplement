import json

with open('./data/train.json') as f:
    train_dict = json.load(f)
    index = 2194
    print(train_dict['answers'][str(index)]['id'])
    print(train_dict['answers'][str(index)]['text'])
