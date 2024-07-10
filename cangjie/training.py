import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from conf import global_settings as settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR

def create_dataloader():
    pass
def train():
    pass

def evaluate():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-batch', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warmup', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    net = get_network(args)

