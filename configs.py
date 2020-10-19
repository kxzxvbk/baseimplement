from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='args for ConvNet')
    parser.add_argument('--output_channel', dest='output_channel', default=32)
    parser.add_argument('--epochs', dest='epochs', default=30)
    parser.add_argument('--batch_size', dest='batch_size', default=128)
    parser.add_argument('--min_length', dest='min_length', default=10)
    parser.add_argument('--resume', dest='resume', default=True)
    parser.add_argument('--use_cuda', dest='use_cuda', default=False)

    args = parser.parse_args()
    return args
