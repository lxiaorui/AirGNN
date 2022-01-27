
import torch
import optuna
import torch_geometric.transforms as T
import argparse
from datasets import str2bool
from train_eval import run
from datasets import prepare_data
from model import AirGNN

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--seed', type=int, default=15)
parser.add_argument('--model', type=str, default='AirGNN')
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--lambda_amp', type=float, default=0.1)
parser.add_argument('--lcc', type=str2bool, default=False)
parser.add_argument('--normalize_features', type=str2bool, default=True)
parser.add_argument('--random_splits', type=str2bool, default=False)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=100)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.8, help="dropout")
parser.add_argument('--K', type=int, default=10, help="the number of propagagtion in AirGNN")
parser.add_argument('--model_cache', type=str2bool, default=False)


def main():
    args = parser.parse_args()
    print('arg : ', args)
    dataset, permute_masks = prepare_data(args, lcc=args.lcc)
    model = AirGNN(dataset, args)
    test_acc = run(dataset, model, args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping,
                   permute_masks, logger=None, args=args) ## TODO: test or val acc
    return test_acc


if __name__ == "__main__":
    main()
