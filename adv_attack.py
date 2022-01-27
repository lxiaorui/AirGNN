from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.targeted_attack import Nettack
import pickle
import argparse
import torch
import numpy as np
from tqdm import tqdm
from datasets import uniqueId, str2bool

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

def cache_nettack_feature_attack_data(args):
    n_perturbations_candidates = [0, 1, 2, 5, 10, 20, 50, 80]
    print("====preparing dataset: %s=====" % (args.dataset))
    cache_dataset(args.dataset, n_perturbations_candidates, args)


def cache_dataset(dataset_name, n_perturbations_candidates, args):
    data = Dataset(root='/tmp/', name=(dataset_name).lower(), seed=args.seed)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device=device)
    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, idx_train)

    node_list = select_nodes(data)
    print(data, node_list, args)

    for target_node in tqdm(node_list):
        for perturbation in n_perturbations_candidates:

            uid = uniqueId(dataset_name, target_node, perturbation)
            n_perturbations = int(perturbation)
            if n_perturbations != 0:
                model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=False,
                                attack_features=True, device=device)
                model = model.to(device)
                model.attack(features, adj, labels, target_node, n_perturbations, verbose=False)
                modified_features = model.modified_features
                data.features = modified_features

            pickle.dump(data, open("./fixed_data/adv_attack/" + uid + ".pickle", 'wb'))
            print(uid, "has been save")

def classification_margin(output, true_label):
    """Calculate classification margin for outputs.
    `probs_true_label - probs_best_second_class`
    Parameters
    ----------
    output: torch.Tensor
        output vector (1 dimension)
    true_label: int
        true label for this node
    Returns
    -------
    list
        classification margin for this node
    """
    probs = torch.exp(output)
    probs_true_label = probs[true_label].clone()
    probs[true_label] = 0
    probs_best_second_class = probs[probs.argmax()]
    return (probs_true_label - probs_best_second_class).item()


def select_nodes(data, target_gcn=None):
    '''
    selecting nodes as reported in nettack paper:
    (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    (ii) the 10 nodes with lowest margin (but still correctly classified) and
    (iii) 20 more nodes randomly
    '''
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    if target_gcn is None:
        target_gcn = GCN(nfeat=features.shape[1],
                         nhid=16,
                         nclass=labels.max().item() + 1,
                         dropout=0.5, device=device)
        target_gcn = target_gcn.to(device)
        target_gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
    target_gcn.eval()
    output = target_gcn.predict()

    margin_dict = {}
    for idx in idx_test:
        margin = classification_margin(output[idx], labels[idx])
        if margin < 0:  # only keep the nodes correctly classified
            continue
        margin_dict[idx] = margin
    sorted_margins = sorted(margin_dict.items(), key=lambda x: x[1], reverse=True)
    high = [x for x, y in sorted_margins[: 10]]
    low = [x for x, y in sorted_margins[-10:]]
    other = [x for x, y in sorted_margins[10: -10]]
    other = np.random.choice(other, 20, replace=False).tolist()

    return high + low + other


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--seed', type=int, default=15)
    args = parser.parse_args()
    print('arg : ', args)
    cache_nettack_feature_attack_data(args)


if __name__ == "__main__":
    main()
    
