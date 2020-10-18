import json


q_filename = '../data/train.query.tsv'
a_filename = '../data/train.reply.tsv'
if __name__ == '__main__':
    '''
    生成对应tsv文件的json标注集，标注集整体的结构如下：
    {
    "length": integer -> 总的样本数量
    "answers": {"id" -> 回答序号: {"id": integer -> 对应的问题id, "text": string -> 问题对应文本} .... }
    "questions": {"id" -> 问题序号: {"text": string -> 问题对应文本} .... }
    }
    '''
    dicts = {}
    que = {}
    ans = {}
    with open(q_filename, encoding='utf-8') as f1:
        line = f1.readline().strip()
        while line is not None:
            try:
                line = line.split('\t')
            except AttributeError:
                continue
            if len(line) != 2:
                break
            id_q, q = line[0], line[1]
            if len(q) == 0:
                line = f1.readline().strip()
                continue
            que[id_q] = {'text': q}
            line = f1.readline().strip()

    temp = 0
    with open(a_filename, encoding='utf-8') as f2:
        line = f2.readline().strip()
        while line is not None:
            try:
                line = line.split('\t')
            except AttributeError:
                continue
            if len(line) != 4:
                break
            id_a, _, a, res = line[0], line[1], line[2], line[3]
            if len(a) == 0:
                line = f2.readline().strip()
                continue
            ans[temp] = {'id': id_a, 'text': a, 'res': res}
            temp += 1
            line = f2.readline().strip()

    dicts['questions'] = que
    dicts['answers'] = ans
    dicts['length'] = temp

    with open('../data/train.json', 'w') as f:
        print('Writing json to: ' + '../data/train.json')
        json.dump(dicts, f)
