import json
import numpy as np
import jieba
import matplotlib.pyplot as plt
import re
import os

# 常用词记录在common_dict.txt里
jieba.load_userdict('./data/common_dict.txt')


def cut_sentence(sentence):
    return list(jieba.cut(sentence))


def cut_list(lis):
    """
    将输入的句子列表，转化成所有句子中出现过的词列表
    重复出现的词不删除
    :param lis: 输入的句子集合
    :return: 分词后的结果
    """
    ans = []
    for sentence in lis:
        ans += cut_sentence(sentence)
    return ans


def handle_one_sentence(sentence):
    # 预处理
    # 去除?, 。, ！
    # 将所有数字替换为 'NUM'
    sentence = sentence.replace('？', '')
    sentence = sentence.replace('?', '')
    sentence = sentence.replace('。', '')
    sentence = sentence.replace('！', '')
    sentence = re.sub(r'\d+', 'NUM', sentence)
    return sentence


def read_corpus():
    """
    读取给定的语料库，并把问题列表和答案列表分别写入到 qlist, alist ,rlist里面。
    在加入过程中调用handle_one_sentence进行预处理
    qlist = ["问题1"， “问题2”， “问题3” ....]
    alist = ["回答1", "回答2", "回答3" ....]
    rlistt = ["结果1", "结果2", "结果3" ....]
    务必要让每一个问题和答案对应起来（下标位置一致）
    """
    qlist = []
    alist = []
    rlist = []

    filename = './data/train.json'
    with open(filename, 'r') as load_f:
        load_dict = json.load(load_f)
        length = load_dict['length']
        # print(load_dict['answers'])
        for i in range(length):
            qlist.append(
                handle_one_sentence(load_dict['questions'][load_dict['answers'][str(i)]['id']]['text'])
            )
            alist.append(
                handle_one_sentence(load_dict['answers'][str(i)]['text'])
            )
            rlist.append(load_dict['answers'][str(i)]['res'])

    return qlist, alist, np.array(rlist)


def visual(word_total):
    # visualization
    # 统计词频

    word_total_old = [word for word in word_total]
    word_total_unique = list(set(word_total))
    # print(word_total_unique)
    # 统计词频
    dict_word_count = {l: 0 for l in word_total_unique}
    for value in word_total:
        dict_word_count[value] += 1

    # 统计出现1,2,3...n的单词的个数
    word_count_set = sorted(list(set(dict_word_count.values())))
    dict_appear_counts = {s: 0 for s in word_count_set}
    for item in dict_word_count.items():
        dict_appear_counts[item[1]] += 1

    x_data = list(dict_appear_counts.keys())
    y_data = list(dict_appear_counts.values())

    fig = plt.figure()  # 设置画布
    ax1 = fig.add_subplot(111)
    k = 50
    plt.plot(x_data[:k], y_data[:k])
    ax1.set_xlabel(u'Word Appear Nums')
    ax1.set_ylabel(u'Word Counts')
    plt.show()

    fig = plt.figure()  # 设置画布
    ax1 = fig.add_subplot(111)
    ax1.hist(x_data, range=(0, 2000), bins=30)
    plt.show()


def words_filter(word_total):
    """
    将输入去重，并且过滤出现次数 < threshold 的单词
    :param word_total: 含重复元素的词集
    :return: 不含重复元素的，经过过滤的词集
    """
    threshold = 3
    word_total_unique = list(set(word_total))
    dict_word_count = {l: 0 for l in word_total_unique}
    for value in word_total:
        dict_word_count[value] += 1
    ans = []
    for val in word_total_unique:
        if dict_word_count[val] >= threshold:
            ans.append(val)
    return ans


def dump_voc_tab(voc_tab):
    diction = {voc_tab[l]: l + 1 for l in range(len(voc_tab))}
    with open('./data/voc_tab.json', 'w') as f:
        json.dump(diction, f)
    print('Vocabulary tab created! Total number: ' + str(len(voc_tab)))
    print('Dump to file: ./data/voc_tab.json')


def load_voc_tab(path='./data/voc_tab.json'):
    print('Loading voc tab from existing file: ' + path)
    with open(path) as f:
        return json.load(f)


if not os.path.exists('./data/voc_tab.json'):
    qlist, alist, _ = read_corpus()
    word_total = [word for word in cut_list(qlist + alist)]
    word_total = words_filter(word_total)
    dump_voc_tab(word_total)
