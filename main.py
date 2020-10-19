import src.train_cnn_net as cnn_net
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


if __name__ == '__main__':
    cnn_net.train()
