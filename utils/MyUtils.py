import os
import torch
import dgl
import matplotlib.pyplot as plt

import GPUtil

class argVar:
    def __init__(self):
        self.dataset='yelp'
        self.train_ratio=0.4
        self.num_classes=2
        self.nodes_per_subgraph=32
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.SSupGCL_num_train_part=20
        self.SSupGCL_batch_size=5

        self.GuiDDPM_train_diffusion_steps=1000
        self.GuiDDPM_train_diffusion_batch_size=100
        self.GuiDDPM_sample_diffusion_steps=100
        self.GuiDDPM_sample_diffusion_batch_size=128
        self.GuiDDPM_sample_guidance_scale=10
        
        self.WFusion_hid_dim=128
        self.WFusion_order=5
        self.WFusion_epochs=250
        self.WFusion_gdc_syn_avg_degree=[]
        self.WFusion_gdc_raw_avg_degree=[]
        self.WFusion_relation_index=[0,1]

        self.SSupGCL_epochs=50 ###
        self.SSupGCL_train_flag=True ###
        self.GuiDDPM_train_steps=6000 ###
        self.GuiDDPM_sample_with_guidance=True ###
        self.WFusion_use_WFusion=True ###

        self.SSupGCL_visualize_flag=False ### visualization of SSupGCL
        self.GuiDDPM_train_flag=True ### train or sample



def pyg_data_to_dgl_graph(pyg_data_obj):
    print(pyg_data_obj)

    # 获取边索引
    edge_index = pyg_data_obj.edge_index
    
    # DGL需要的边索引格式是两列的数组，而不是两行的索引
    src, dst = edge_index
    edge_list = torch.stack((src, dst), dim=1)
    
    # 创建DGL图
    g = dgl.graph((edge_list[:, 0], edge_list[:, 1]), num_nodes=pyg_data_obj.x.shape[0])
    
    # 添加节点特征
    if 'x' in pyg_data_obj:
        g.ndata['feature'] = pyg_data_obj.x
    
    # 添加边特征
    if 'edge_attr' in pyg_data_obj:
        g.edata['feat'] = pyg_data_obj.edge_attr
    
    # 添加节点标签
    if 'y' in pyg_data_obj:
        g.ndata['label'] = pyg_data_obj.y

    # 添加节点标签
    if 'train_mask' in pyg_data_obj:
        g.ndata['train_mask'] = pyg_data_obj.train_mask

    # 添加节点标签
    if 'test_mask' in pyg_data_obj:
        g.ndata['test_mask'] = pyg_data_obj.test_mask

    # 添加节点标签
    if 'val_mask' in pyg_data_obj:
        g.ndata['val_mask'] = pyg_data_obj.val_mask

    
    return g

def get_gpu_memory_usage():
    # 获取GPU使用情况
    gpus = GPUtil.getGPUs()
    gpu_memory_info = []
    for gpu in gpus:
        gpu_memory_info.append((gpu.id, gpu.name, f"{gpu.memoryUsed} MB", f"{gpu.memoryTotal} MB", f"{gpu.memoryUtil * 100:.1f}%"))

    return gpu_memory_info

def display_gpu_memory_usage():
    gpu_memory_info = get_gpu_memory_usage()

    if gpu_memory_info:
        print("GPU Memory Usage:")
        for info in gpu_memory_info:
            print(f"GPU ID: {info[0]}, Name: {info[1]}, Memory Used: {info[2]}, Total Memory: {info[3]}, Memory Utilization: {info[4]}")
    else:
        print("No GPU found.")

def color_print(content):
    print(f'\033[1;46m{content}\033[0m\n')

def save_pic_iterly(pic_name, postfix, info):
    pic_idx=1
    pic_name_full=f'{pic_name}_{pic_idx}.{postfix}'

    while os.path.exists(pic_name_full):
        print(f'File {pic_name_full} already exists.')
        pic_idx += 1
        pic_name_full=f'{pic_name}_{pic_idx}.png'

    plt.savefig(pic_name_full, dpi=300, bbox_inches='tight')

    color_print(f'!!!!! {info} is saved in file {pic_name_full}')
