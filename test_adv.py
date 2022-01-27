import torch
import argparse
import os
import pickle
from datasets import prepare_data
from model import AirGNN
from tqdm import tqdm
from train_eval import evaluate
import numpy as np
from torch import tensor
from datasets import uniqueId, str2bool
import torch_geometric.transforms as T
from deeprobust.graph.data import Dpr2Pyg


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--model', type=str, default='AirGNN')
parser.add_argument('--normalize_features', type=str2bool, default=True)
parser.add_argument('--random_splits', type=str2bool, default=False)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--dropout', type=float, default=0.5, help="dropout")
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--K', type=int, default=10, help="the number of propagagtion in AirGNN")
parser.add_argument('--alpha', type=float, default=None)
parser.add_argument('--lambda_amp', type=float, default=0.1)


args = parser.parse_args()
print('arg : ', args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: in train_eval:", device)


def main():
    acc_lst_dic = {}
    final_result = {}
    n_perturbations_candidates = [0, 1, 2, 5, 10, 20, 50, 80]
    for perturbation_number in n_perturbations_candidates:
        acc_lst_dic[perturbation_number] = []

    if args.dataset in ["Cora", "CiteSeer", "PubMed"]:
        for run_k in range(args.runs):
            perturbation_acc_dic = nettack_run(args.dataset.lower(), run_k, n_perturbations_candidates)
            for key, val in perturbation_acc_dic.items():
                acc_lst_dic[key].append(val)

        for key, val in acc_lst_dic.items():
            acc_lst = tensor(val)
            final_result[key] = '{:.3f} ± {:.3f}'.format(acc_lst.mean().item(), acc_lst.std().item())
            
        print("Dataset:{}, model:{}".format(args.dataset, args.model))
        print("Average performance on 40 targeted nodes with 10 runs:", final_result)

def nettack_run(dataset_name, run_k, n_perturbations_candidates):
    node_list = get_target_nodelst(dataset_name)
    num = len(node_list)
    assert num == 40

    target_accuracy_dic = {}
    target_accuracy_summary_dic = {}
    for key in n_perturbations_candidates:
        target_accuracy_dic[key] = 0
        target_accuracy_summary_dic[key] = []

    for target_node in node_list:
        for perturbation in n_perturbations_candidates:
            n_perturbations = int(perturbation)
            uid = uniqueId(dataset_name, target_node, perturbation)
            data = get_adv_data(uid)
            target_node_acc = adv_test(target_node, data, data.adj[target_node].nonzero()[1].tolist(), run_k)
            if target_node_acc == 0:
                target_accuracy_dic[perturbation] += 1

            print("=========Attacked Node: {:d}, n_perturbations: {:.2f}=========".format(target_node, perturbation))
        print(args.model, args.lambda_amp)

    assert num == 40
    for key in target_accuracy_dic.keys():
        target_accuracy_dic[key] = 1 - target_accuracy_dic[key] / num

    print("Accuracy on 40 target nodes:", target_accuracy_dic)
    return target_accuracy_dic


def get_target_nodelst(dataset_name):
    allfiles = os.listdir("./fixed_data/adv_attack")
    remain = [single_file for single_file in allfiles if dataset_name.lower() in single_file]
    node_indexes = []
    for i in remain:
        if len(i.split("_"))  == 3:
            node_indexes.append(int(i.split("_")[1]))

    node_list = list(set(node_indexes))
    assert len(node_list) == 40
    return node_list

def get_adv_data(uid):
    print("./fixed_data/adv_attack/"+uid+".pickle")
    if os.path.isfile("./fixed_data/adv_attack/"+uid+".pickle"):
        return pickle.load(open("./fixed_data/adv_attack/"+uid+".pickle",'rb'))
    else:
        raise Exception("ERROR" + uid + " file not found")

def adv_test(key_node_index, attacked_dpr_data, neighbor_lst, run_k):
    transform = T.ToSparseTensor()
    dataset = Dpr2Pyg(attacked_dpr_data, transform=transform)
    data = dataset[0]
    data = data.to(device)

    if args.model in ["APPNP", "AirGNN", "MLP"]:
        model = AirGNN(dataset, args)
    else:
        raise Exception("Unsupported model mode!!!")

    # 10 best models will be tested on the same attacked data
    model.to(device).reset_parameters()
    checkpointPath = "./model/lcc/{}_{}_best_model_run_{}.pth".format(args.dataset, args.model, run_k)
    print("checkpointPath:", checkpointPath)
    checkpoint = torch.load(checkpointPath)
    model.load_state_dict(checkpoint["model_state_dict"])

    # test the prediction of targeted node
    model.eval()
    logits = model(data)
    probs = torch.exp(logits[[key_node_index]])
    target_node_acc = (logits.argmax(1)[key_node_index] == data.y[key_node_index]).item()  # True/False
    # print("single_target_predict\n:", logits.argmax(1)[key_node_index].item(), data.y[key_node_index].item(), target_node_acc)
    return target_node_acc



if __name__ == "__main__":
    main()
