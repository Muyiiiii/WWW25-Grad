import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.transforms import GDC
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from scipy.io import loadmat
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import scipy.io
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import networkx as nx

import numpy as np

from tqdm import tqdm

import argparse
import os

from utils.dataProcess import loadDataset, mergeGraphDataList, GDCAugment, data4WFusionTrain
from utils.MyUtils import color_print, argVar

from models.WeightedFusion import WeightFusion, WFusionTrain


def parse_args():
    parser = argparse.ArgumentParser(description='BWGNN')
    parser.add_argument("--dataset", type=str, default="yelp",
                        help="Dataset for this model (yelp/amazon/tfinance/tsocial)")
    parser.add_argument("--train_ratio", type=float, default=0.4, help="Training ratio")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
    parser.add_argument("--homo", type=int, default=0, help="1 for BWGNN(Homo) and 0 for BWGNN(Hetero)")
    parser.add_argument("--epoch", type=int, default=250, help="The max number of epochs")
    parser.add_argument("--run", type=int, default=1, help="Running times")
    parser.add_argument("--dataset", type=str, default="yelp", help="Dataset name")
    parser.add_argument("--train_ratio", type=float, default=0.4, help="Training ratio")
    parser.add_argument("--nodes_per_subgraph", type=int, default=32, help="Number of nodes per subgraph")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use for computation")
    parser.add_argument("--SSupGCL_train_flag", type=bool, default=True, help="Flag for SSupGCL training")
    parser.add_argument("--SSupGCL_num_train_part", type=int, default=20, help="Number of training partitions for SSupGCL")
    parser.add_argument("--SSupGCL_batch_size", type=int, default=5, help="Batch size for SSupGCL training")
    parser.add_argument("--SSupGCL_epochs", type=int, default=100, help="Number of epochs for SSupGCL training")
    parser.add_argument("--SSupGCL_visualize_flag", type=bool, default=True, help="Flag for visualizing SSupGCL results")
    parser.add_argument("--GuiDDPM_train_flag", type=bool, default=True, help="Flag for GuiDDPM training")
    parser.add_argument("--GuiDDPM_train_steps", type=int, default=3000, help="Number of training steps for GuiDDPM")
    parser.add_argument("--GuiDDPM_train_diffusion_steps", type=int, default=1000, help="Number of diffusion steps during GuiDDPM training")
    parser.add_argument("--GuiDDPM_sample_diffusion_steps", type=int, default=1000, help="Number of diffusion steps during GuiDDPM sampling")

    args = parser.parse_args()
    
    return args

def main():
    final_ap=[]
    final_auc=[]
    for i in tqdm([1]):
        args = argVar()
        # print(args)
        # prepare data
        graph_dgl, graph_pyg, train_mask, val_mask, test_mask = loadDataset(dataset=args.dataset, train_ratio=args.train_ratio)
        in_feats = graph_dgl.ndata['feature'].shape[1]
        num_classes = 2

        if args.GuiDDPM_sample_with_guidance:
            syn_relation_filename = f"./Generation/SynRelation_{args.dataset}_{args.GuiDDPM_sample_diffusion_steps}Samplesteps_{args.GuiDDPM_train_steps}Trainsteps_subgraphsize_{args.nodes_per_subgraph}_guided.pt"
        else:
            syn_relation_filename = f"./Generation/SynRelation_{args.dataset}_{args.GuiDDPM_sample_diffusion_steps}Samplesteps_{args.GuiDDPM_train_steps}Trainsteps_subgraphsize_{args.nodes_per_subgraph}_unguided.pt"

        # syn_relation_filename='./Generation/ddpm_yelp_dict_even-v2.pt'

        syn_relation_dict=torch.load(syn_relation_filename)

        graph_syn=mergeGraphDataList(args=args, graph_pyg=graph_pyg, syn_relation_dict=syn_relation_dict)

        print(graph_syn)

        color_print(f'!!!!! Strat gdc augment')
        graph_gdc_list=[]
        for avg_degree in tqdm(args.WFusion_gdc_syn_avg_degree):
            filename=f'./Generation/GDCAugGraph/GDC_SynRelation_{args.dataset}_{args.GuiDDPM_sample_diffusion_steps}Samplesteps_{args.GuiDDPM_train_steps}Trainsteps_avgdegree_{avg_degree}.pt'

            if os.path.exists(filename):
                gdc_aug_graph=torch.load(filename)
                color_print(f'gdc_aud{len(graph_gdc_list)+1} is load from {filename}') 
            else:
                gdc_aug_graph=GDCAugment(graph_pyg_type=graph_syn, avg_degree=avg_degree)
                torch.save(gdc_aug_graph,filename)
                color_print(f'gdc_aud{len(graph_gdc_list)+1} is saved in {filename}') 
            graph_gdc_list.append(gdc_aug_graph)

        for avg_degree in tqdm(args.WFusion_gdc_raw_avg_degree):
            filename=f'./Generation/GDCAugGraph/GDC_RawRelation_{args.dataset}_{args.GuiDDPM_sample_diffusion_steps}Samplesteps_{args.GuiDDPM_train_steps}Trainsteps_avgdegree_{avg_degree}.pt'

            if os.path.exists(filename):
                gdc_aug_graph=torch.load(filename)
                color_print(f'gdc_aud{len(graph_gdc_list)+1} is load from {filename}') 
            else:
                gdc_aug_graph=GDCAugment(graph_pyg_type=graph_pyg, avg_degree=avg_degree)
                torch.save(gdc_aug_graph,filename)
                color_print(f'gdc_aud{len(graph_gdc_list)+1} is saved in {filename}') 
            graph_gdc_list.append(gdc_aug_graph)
            
        color_print(f'!!!!! Finish gdc augment')

        graph_WFusion=data4WFusionTrain(graph_pyg=graph_pyg, 
                                        graph_syn=graph_syn, 
                                        graph_gdc_list=graph_gdc_list
                                        )

        print(f"{graph_WFusion.ndata['feature'].shape}")

        model_WFusion=WeightFusion(global_args=args, in_feats=graph_WFusion.ndata['feature'].shape[1], h_feats=args.WFusion_hid_dim, num_classes=num_classes, graph=graph_WFusion, d=args.WFusion_order, relations_idx=args.WFusion_relation_index, device=args.device).to(args.device)

        auc,ap,losses_2,auc_2=WFusionTrain(model_WFusion, graph_WFusion, args,graph_WFusion.ndata['train_mask'],graph_WFusion.ndata['val_mask'],graph_WFusion.ndata['test_mask'])
        final_ap.append(ap)
        final_auc.append(auc)

    # color_print(f'auc:{final_auc}')
    # color_print(f'ap:{final_ap}')





if __name__=='__main__':
    main()
