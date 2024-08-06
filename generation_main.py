from utils.MyUtils import color_print, argVar
from utils.dataProcess import loadDataset, nodeSelect, nodeSample, sampleCheck
from models.Semi_SupGCL import SSupGCL
# my part
from models.GuiDDPM import GuiDDPM

import torch

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameters configuration")
    
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

def SSupGCL_module(args, graph_pyg, graph_pyg_selected):
    model_SSupGCL=SSupGCL(global_args=args,
                          graph_pyg_selected=graph_pyg_selected,
                          nodes_per_subgraph=args.nodes_per_subgraph,
                          device=args.device,
                          num_train_part=args.SSupGCL_num_train_part,
                          batch_size=args.SSupGCL_batch_size)
    
    SSupGCL_para_filename=f'./ModelPara/SSupGCLPara/SSupGCL_{args.dataset}_{args.SSupGCL_epochs}epochs_subgraphsize_{args.nodes_per_subgraph}.pt'

    if args.SSupGCL_train_flag and not os.path.exists(SSupGCL_para_filename):
        model_SSupGCL.train(epochs=args.SSupGCL_epochs)
        model_SSupGCL.save_model(path=SSupGCL_para_filename)
    else:
        model_SSupGCL.load_model(path=SSupGCL_para_filename)
    
    if args.SSupGCL_visualize_flag:
        model_SSupGCL.visualize()

    return model_SSupGCL.project(graph_pyg=graph_pyg), model_SSupGCL

def GuiDDPM_module(args, graph_pyg_ssupgcl, node_groups, edge_index_unselected, guidance):
    GuiDDPM_para_filename = f"./ModelPara/GuiDDPMPara/GuiDDPM_{args.dataset}_{args.GuiDDPM_train_steps}steps_subgraphsize_{args.nodes_per_subgraph}.pt"
    if args.GuiDDPM_sample_with_guidance:
        syn_relation_filename = f"./Generation/SynRelation_{args.dataset}_{args.GuiDDPM_sample_diffusion_steps}Samplesteps_{args.GuiDDPM_train_steps}Trainsteps_subgraphsize_{args.nodes_per_subgraph}_guided.pt"
    else:
        syn_relation_filename = f"./Generation/SynRelation_{args.dataset}_{args.GuiDDPM_sample_diffusion_steps}Samplesteps_{args.GuiDDPM_train_steps}Trainsteps_subgraphsize_{args.nodes_per_subgraph}_unguided.pt"

    model_DDPM=GuiDDPM(global_args=args,
                      graph_pyg_ssupgcl=graph_pyg_ssupgcl,
                      node_groups=node_groups, 
                      edge_index_unselected=edge_index_unselected,
                      guidance=guidance, 
                      train_flag=args.GuiDDPM_train_flag, 
                      model_path=GuiDDPM_para_filename,
                      syn_relation_filename=syn_relation_filename,
                      device=args.device)

    # if args.GuiDDPM_train_flag and not os.path.exists(GuiDDPM_para_filename):
    if args.GuiDDPM_train_flag:
        if os.path.exists(GuiDDPM_para_filename):
            color_print(f'!!!!! GuiDDPM Parameter is loaded from {GuiDDPM_para_filename} Success')
        else:
            model_DDPM.train(train_steps=args.GuiDDPM_train_steps)
            model_DDPM.save_model(GuiDDPM_para_filename)
    else:
        model_DDPM.sample()

def main():
    args=argVar()
    
    print(f'device: {args.device}')

    # prepare data
    graph_dgl, graph_pyg, train_mask, val_mask, test_mask = loadDataset(dataset=args.dataset, train_ratio=args.train_ratio)

    edge_types = graph_dgl.canonical_etypes

    print(graph_pyg.y[train_mask].sum(),graph_pyg.y[val_mask].sum(),graph_pyg.y[test_mask].sum())

    graph_pyg_selected, edge_index_unselected=nodeSelect(graph_pyg=graph_pyg, nodes_per_subgraph=32)

    # Semi-supervised graph contrastive learning (SSupGCL)
    graph_pyg_ssupgcl, model_SSupGCL=SSupGCL_module(args=args,
                                            graph_pyg=graph_pyg,
                                            graph_pyg_selected=graph_pyg_selected)

    # node sample
    node_groups=nodeSample(graph_pyg=graph_pyg_ssupgcl, nodes_per_subgraph=args.nodes_per_subgraph)
    sampleCheck(node_groups=node_groups, nodes_per_subgraph=args.nodes_per_subgraph)

    # GuiDDPM
    GuiDDPM_module(args=args, 
                   graph_pyg_ssupgcl=graph_pyg_ssupgcl,
                   node_groups=node_groups,
                   edge_index_unselected=edge_index_unselected, 
                   guidance=model_SSupGCL)
    
    # Weighted Filter
    



if __name__=='__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()
