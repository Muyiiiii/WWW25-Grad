# Grad: Guided Relation Diffusion Generation for Graph Augmentation in Graph Fraud Detection



## Requirements

This code requires the following:

- python==3.9

- pytorch==1.12.1+cu113

  - ```
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
    ```

  - ```
    pip uninstall numpy
    ```

  - ```
    pip install numpy==1.26.0
    ```

- dgl==0.9.1+cu113

  - ```
    pip install dgl-cu113 -f https://data.dgl.ai/wheels/repo.html
    ```

- pygcl

  - ```
    pip install PyGCL
    ```

- mpi4py

  - ```
    conda install mpi4py
    ```


    - or
      
    - ```
      pip install mpi4py
      ```

- if error in pip


    - ```
      apt-get update
      ```
        

    - ```
      apt-get install mpich
      ```


    - ```
      pip install mpi4py
      ```


- torch_geometric==2.2.0

- improved-diffusion

  - ```
    cd models
    ```

  - ```
    pip install -e .
    ```



## Usage

**The args are in the utils/MyUtils.py**

```python
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
```

1. **Train diffusion**
   Can set the `self.GuiDDPM_train_steps` for your like.

   ```python
   self.GuiDDPM_train_flag=True
   ```

   ```
   python generation_main.py
   ```

2. **Sample with guidance**

   ```python
   self.GuiDDPM_train_flag=False
   ```

   ```
   python generation_main.py
   ```

3. **Detection**

   ```
   python generation_main.py
   ```

   
