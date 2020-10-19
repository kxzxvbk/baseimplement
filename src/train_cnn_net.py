import utils.prepocessing
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from utils.checkpoint import Checkpoint
from utils.optim import NoamOpt
from libs import models
import torch
import libs.dataset
import configs

"""
问题：存在非常明显的局部最优现象，基本所有的预测都是0
可能的解决方案：
   调整embedding的dim
   调整output_channel
   调整loss，在loss设置中，我把对类别1的惩罚调到了4倍
   加长训练时间
如果都没的救，可能只能换model了
"""


def test(dtloader, mdl, epo_id):
    """
    :param dtloader: dataloader
    :param mdl: 用于测试的模型
    :param epo_id: 迭代次数id
    :return: f1(比赛评价标准)
    """
    acc = 0
    total_acc2 = 0
    total_acc1 = 0
    print('Testing model in EPOCH {}'.format(epo_id))
    for _, dt in enumerate(dtloader):
        xx = dt['que'].long()
        yy = dt['ans'].long()
        ans = dt['res'].float()
        ans_pred = mdl(xx, yy).squeeze(0)
        if ans_pred[1] >= ans_pred[0] and ans == 1:
            acc += 1
        if ans == 1:
            total_acc2 += 1
        if ans_pred[1] >= ans_pred[0]:
            total_acc1 += 1

    acc_rate = (acc + 1) / (total_acc1 + 1)
    recall_rate = (acc + 1) / (total_acc2 + 1)
    f1 = 2 / (1 / acc_rate + 1 / recall_rate)
    return f1


def train():
    args = configs.get_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()

    # prepare dataset
    dataset = libs.dataset.MyDataset(min_length=args.min_length)
    voc_size = dataset.get_voc_size()
    dataloader = DataLoader(dataset, 1, True, drop_last=False)

    # prepare model
    model = models.TopModuleCNN(voc_size, output_channel=args.output_channel)
    if use_cuda:
        model = model.cuda()

    # load pretrained if asked
    if args.resume:
        checkpoint_path = Checkpoint.get_certain_checkpoint("./experiment/cnn_net", "best")
        resume_checkpoint = Checkpoint.load(checkpoint_path)
        model = resume_checkpoint.model
        optimizer = resume_checkpoint.optimizer

        resume_optim = optimizer.optimizer
        defaults = resume_optim.param_groups[0]
        defaults.pop('params', None)
        optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

        start_epoch = resume_checkpoint.epoch
        max_ans_acc = resume_checkpoint.max_ans_acc
    else:
        start_epoch = 1
        max_ans_acc = 0
        optimizer = NoamOpt(512, 1, 2000,
                            optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    # define loss
    loss = nn.CrossEntropyLoss(weight=torch.tensor([1., 4.]))
    if use_cuda:
        loss = loss.cuda()

    # training
    for i in range(start_epoch, args.epochs):
        # test the model
        if args.resume:
            test_ans_acc = max_ans_acc
        else:
            test_ans_acc = test(DataLoader(dataset, 1, True, drop_last=False), model, i)
        print('For EPOCH {}, total f1: {:.2f}'.format(i, test_ans_acc))

        # calculate loss
        j = 0
        los1 = []
        for _, data in enumerate(dataloader):
            j += 1
            x = data['que'].long()
            y = data['ans'].long()
            res = data['res'].long()
            if use_cuda:
                x, y, res = x.cuda(), y.cuda(), res.cuda()
            res_pred = model(x, y)

            los1.append(loss(res_pred, res).unsqueeze(0))

            # apply gradient
            if j % args.batch_size == 0:
                los1 = torch.cat(los1)
                los = los1.sum()
                model.zero_grad()
                los.backward()
                optimizer.step()
                los1 = []
                print('EPOCH: {}, {} / {}====> LOSS: {:.2f}'
                      .format(i, j // args.batch_size, dataloader.__len__() // args.batch_size,
                              los.item() / args.batch_size))

        # save checkpoint
        if test_ans_acc > max_ans_acc:
            max_ans_acc = test_ans_acc
            th_checkpoint = Checkpoint(model=model,
                                       optimizer=optimizer,
                                       epoch=i,
                                       max_ans_acc=max_ans_acc)
            th_checkpoint.save_according_name("./experiment/cnn_net", 'best')
