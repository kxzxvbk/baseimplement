import numpy as np
from torch.utils.data import Dataset
import json
import jieba


class MyDataset(Dataset):
    def __init__(self, min_length=None, train_json='./data/train.json', voc_tab='./data/voc_tab.json'):
        self.train_json = train_json
        self.voc_tab = voc_tab
        self.min_length = min_length
        with open(self.train_json, encoding='utf-8') as f:
            self.train_dict = json.load(f)
        with open(self.voc_tab, encoding='utf-8') as f:
            self.voc_dict = json.load(f)
        self.voc_size = len(self.voc_dict) + 1

    def get_voc_size(self):
        return self.voc_size

    def __getitem__(self, index):
        que = self.train_dict['questions'][self.train_dict['answers'][str(index)]['id']]['text']
        ans = self.train_dict['answers'][str(index)]['text']
        res = self.train_dict['answers'][str(index)]['res']

        que = list(jieba.cut(que))
        ans = list(jieba.cut(ans))

        if self.min_length is not None:
            que_encode = np.zeros([max(len(que), self.min_length)])
            ans_encode = np.zeros([max(len(ans), self.min_length)])
        else:
            que_encode = np.zeros([len(que)])
            ans_encode = np.zeros([len(ans)])

        for i in range(len(que)):
            if que[i] in self.voc_dict:
                que_encode[i] = int(self.voc_dict[que[i]])
            else:
                que_encode[i] = len(self.voc_dict) - 1

        for j in range(len(ans)):
            if ans[j] in self.voc_dict:
                ans_encode[j] = int(self.voc_dict[ans[j]])
            else:
                ans_encode[j] = len(self.voc_dict) - 1

        if len(que) == 0:
            print(self.train_dict['questions'][self.train_dict['answers'][str(index)]['id']]['text'])
            print(index)
        if len(ans) == 0:
            print(self.train_dict['answers'][str(index)]['text'])
            print(index)
        return {'que': que_encode, 'ans': ans_encode, 'res': int(res)}

    def __len__(self):
        return self.train_dict['length']

