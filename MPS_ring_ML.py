from utils import way_for_loop_2,way_for_efficient_coding_einsum,way_for_loop_einsum,contract_physics_bond_first
from utils import get_BatchedTensorNetwork
from model_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import math
class MPS_Ring(nn.Module):
    def __init__(self, nums=20, out_class = 10,
                       in_physics_bond = 2, virtual_bond_dim=2,
                       bias=True,init_std=1e-10,**kargs):
        super().__init__()
        self.vd  = vd      = virtual_bond_dim
        self.pd  = pd      =  in_physics_bond
        self.cd  = cd      = out_class
        self.nums          = nums
        self.left          = nn.Parameter(self.rde1((nums-1,vd,pd,vd),init_std), requires_grad=True)
        self.middle        = nn.Parameter(self.rde1((    cd,vd,pd,vd),init_std), requires_grad=True)

    @staticmethod
    def rde1(shape,init_std):
        if len(shape) == 2:
            a,b = shape
            bias_mat     = torch.eye(a, b)
            tensor       = init_std * torch.randn(a, b)+ bias_mat
        elif len(shape) >= 3:
            a,b,c        = shape[-3:]
            bias_mat     = torch.eye(a, c).unsqueeze(1).repeat(1,b,1)
            tensor       = init_std * torch.randn(*shape)+ bias_mat
        #tensor/=tensor.norm()
        return tensor

    def forward(self, input_data):
        left_tensors = torch.einsum("lisj,bls->lbij", self.left  , input_data[:, :-1])
        rigt_tensors = torch.einsum("cisj,bs ->bcij", self.middle, input_data[:,  -1])
        left_tensors = self.get_batch_chain_contraction_loop(left_tensors)
        return torch.einsum("bij,bcji->bc", left_tensors, rigt_tensors)
class MPS_ring_LP_method(MPS_Ring):
    @staticmethod
    def get_chain_contraction_loop(tensor):
        size   = len(tensor)
        now_tensor= tensor[0]
        for next_tensor in tensor[1:]:
            now_tensor = torch.einsum("ik,kj->ij",now_tensor, next_tensor)
        return now_tensor

    def forward(self, input_data):
        tensor_list = []
        for data in input_data:
            left_tensors = torch.einsum("lisj,ls->lij", self.left  , data[:-1])
            rigt_tensors = torch.einsum("cisj,s ->cij", self.middle, data[ -1])
            left_tensors = self.get_chain_contraction_loop(left_tensors)
            tensor_list.append(torch.einsum("ij,cji->c", left_tensors, rigt_tensors))
        return torch.stack(tensor_list)

class MPS_ring_EC_method(MPS_Ring):
    @staticmethod
    def get_batch_chain_contraction_loop(tensor):
        now_tensor= tensor[0]
        for next_tensor in tensor[1:]:
            now_tensor = torch.einsum("bik,bkj->bij",now_tensor, next_tensor)
        return now_tensor
    def forward(self, input_data):
        left_tensors = torch.einsum("lisj,bls->lbij", self.left  , input_data[:, :-1])
        rigt_tensors = torch.einsum("cisj,bs ->bcij", self.middle, input_data[:,  -1])
        left_tensors = self.get_batch_chain_contraction_loop(left_tensors)
        return torch.einsum("bij,bcji->bc", left_tensors, rigt_tensors)

class MPS_ring_BTN_method(MPS_Ring):
    def __init__(self,batch_num=None,cal_best_path=False,**kargs):
        assert batch_num is not None
        super().__init__(**kargs)
        tn.set_default_backend('pytorch')
        left_nodes  = [tn.Node(v, name=f"t{i}") for i,v in enumerate(self.left)]
        # (V,k,V)   <->   (V,k,V) <->   (V,k,V) <->  (V,k,V) <-> ...
        for i in range(len(left_nodes)-1):
            tn.connect(left_nodes[i][-1],left_nodes[i+1][0],name=f'mps{i}-mps{i+1}')
        i = len(left_nodes)
        middle_node = tn.Node(self.middle, name=f"mps{i}")
        tn.connect(left_nodes[-1][-1],middle_node[1],name=f'mps{i-1}-mps{i}')
        tn.connect(middle_node[-1],left_nodes[0][0],name=f'mps{i}-mps{0}')

        batch      = get_BatchedTensorNetwork(torch.randn(batch_num,self.nums,self.pd))
        inp_nodes  = [tn.Node(v, name=f"i{i}") for i,v in enumerate(batch)]
        tn.connect(inp_nodes[0][0],inp_nodes[1][0],name=f'inp{0}-inp{1}')
        for i in range(1,len(inp_nodes)-1):
            tn.connect(inp_nodes[i][-1],inp_nodes[i+1][0],name=f'inp{i}-inp{i+1}')
        for i in range(len(left_nodes)):
            tn.connect(left_nodes[i][1],inp_nodes[i][1],name=f'mps{i}-inp{i}')
        i = len(left_nodes)
        tn.connect(middle_node[2],inp_nodes[i][1],name=f'mps{i}-inp{i}')

        node_list = left_nodes+[middle_node]+inp_nodes
        node_list,sublist_list,outlist = get_sublist_from_node_list(node_list,outlist=[inp_nodes[-1][-1],middle_node[0]])
        operands = []
        for node,sublist in zip(node_list,sublist_list):
            operands+=[node.tensor,sublist]
        operands+= [outlist]
        print("get network constructed")
        json_file= 'the_optim_contraction_path_store.json'
        if not os.path.exists(json_file):
            path_pool={}
            cal_best_path = True
        else:
            with open(json_file,'r') as f:
                path_pool = json.load(f)
        shape_array_string = full_size_array_string(*operands)
        if shape_array_string not in path_pool:cal_best_path = True
        if cal_best_path:
            re_saveQ =True
            T_list = [10,1,0.1,0.1]
            for anneal_idx in [0,1,2]:
                optimizer = oe.RandomGreedy(max_time=20, max_repeats=1000)
                for T in T_list[anneal_idx:]:
                    optimizer.temperature = T
                    path_rand_greedy = oe.contract_path(*operands, optimize=optimizer)
                    print(math.log2(optimizer.best['flops']))
                optimizer.best['path']        = oe.paths.ssa_to_linear(optimizer.best['ssa_path'])
                optimizer.best['outlist']     = outlist
                optimizer.best['sublist_list']= sublist_list
            if shape_array_string not in path_pool:
                path_pool[shape_array_string] = optimizer.best
                re_saveQ=True
                print(f"save best:float-({optimizer.best['flops']})")
            else:
                save_best_now = path_pool[shape_array_string]
                if optimizer.best['flops'] < save_best_now['flops']:
                    print(f"old best:float-({save_best_now['flops']})")
                    print(f"now best:float-({optimizer.best['flops']})")
                    re_saveQ=True
                    path_pool[shape_array_string] = optimizer.best
                else:
                    re_saveQ=False
            if re_saveQ:
                with open(json_file,'w') as f:
                    json.dump(path_pool,f)
        path = path_pool[shape_array_string]['path']
        #path,info = oe.contract_path(*operands,optimize='random-greedy-128')
        self.contraction_info = sublist_list,outlist,path

    def forward(self, input_data):
        batch                     = get_BatchedTensorNetwork(input_data)
        sublist_list,outlist,path = self.contraction_info
        tensor_list = [t for t in self.left] +[model.middle]+ [t for t in batch]
        operands=[]
        for tensor,sublist in zip(tensor_list,sublist_list):
            operand = [tensor,[...,*sublist]]
            operands+=operand
        operands+= [[...,*outlist]]
        return oe.contract(*operands,optimize=path)

transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.1307,), (0.3081,)),
    transforms.CenterCrop(24),
    #transforms.Resize(16)
])
DATAPATH    = '/media/tianning/DATA/DATASET/MNIST/'
mnist_train = datasets.MNIST(DATAPATH, train=True, download=False, transform=transform)
mnist_test  = datasets.MNIST(DATAPATH, train=False,download=False, transform=transform)
train_loader= torch.utils.data.DataLoader(dataset=mnist_train, batch_size=100, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test, batch_size =100, shuffle=False)

import time
from mltool.dataaccelerate import DataSimfetcher
from mltool.loggingsystem import LoggingSystem

def preprocess_images(image):
    image = image.reshape(len(image),-1)
    image = image.round().float()
    return torch.stack([1-image,image],-1)

model_list = [#MPS_ring_EC_method,
              MPS_ring_BTN_method,
              #MPS_ring_LP_method
              ]
TIME_NOW    = time.strftime("%m_%d_%H_%M_%S")
vd = 20
lr = 0.0001
epoches = 5
for MODELTYPE in model_list:
    random_seed=1
    model_name= MODELTYPE.__name__

    ROOTDIR = f"checkpoints/MNIST16x16/{model_name}-{random_seed}-vd{vd}-{TIME_NOW}"
    logsys = LoggingSystem(True,ROOTDIR,seed=random_seed)

    model  = MODELTYPE(nums=24*24, out_class = 10,batch_num=100,cal_best_path=False,
                           in_physics_bond = 2, virtual_bond_dim=vd,init_std=1e-2)
    device = 'cuda'
    model  = model.to(device)


    infiniter = DataSimfetcher(train_loader, device=device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    loss_fn   = torch.nn.CrossEntropyLoss()
    metric_list       = ['loss','accu_t','accu']
    losses=[]
    accues=[]
    accu_t=[]
    metric_dict       = logsys.initial_metric_dict(metric_list)
    master_bar        = logsys.create_master_bar(epoches)
    #master_bar.set_multiply_graph(figsize=(9,3),engine=[['plot','plot','plot']],labels=[metric_list])
    accu = loss = best = -1
    for epoch in master_bar:
        start_time = time.time()
        model.train()
        infiniter = DataSimfetcher(train_loader, device=device)

        inter_b   = logsys.create_progress_bar(len(train_loader))
        while inter_b.update_step():
            image,label= infiniter.next()
            binary     = preprocess_images(image)
            logits     = model(binary).squeeze()
            loss       = loss_fn(logits,label.squeeze())
            pred_labels  = torch.argmax(logits,-1)
            acct       =  torch.sum(pred_labels == label)/len(label)
            loss.backward()
            if torch.isnan(loss):raise

            optimizer.step()
            loss = loss.item()
            acct = acct.item()
            losses.append([time.time(),loss])
            accu_t.append([time.time(),acct])
            inter_b.lwrite('Epoch: %.3i \t Train_Loss: %.4f \t Train_Accu: %.4f Valid_Accu: %.4f Valid_best: %.4f \t Time: %.2f s' %(epoch, loss, acct,accu,best,time.time() - start_time),end='\r')
        model.eval()
        prefetcher = DataSimfetcher(test_loader, device=device)
        inter_b    = logsys.create_progress_bar(len(test_loader))
        labels     = []
        logits     = []
        with torch.no_grad():
            while inter_b.update_step():
                image,label= prefetcher.next()
                binary     = preprocess_images(image)
                logit      = model(binary).squeeze()
                loss       = loss_fn(logit ,label.squeeze())
                labels.append(label)
                logits.append(logit)
        labels  = torch.cat(labels)
        logits  = torch.cat(logits)
        pred_labels  = torch.argmax(logits,-1)
        accu =  torch.sum(pred_labels == labels)/len(labels)
        accu = accu.item()
        accues.append([time.time(),accu])
        if accu >best:best = accu
        inter_b.lwrite('Epoch: %.3i \t Train_Loss: %.4f \t Train_Accu: %.4f Valid_Accu: %.4f Valid_best: %.4f \t Time: %.2f s' %(epoch, loss, acct,accu,best,time.time() - start_time),end='\r')
    import os
    project_info = {}
    project_info['lr'] = lr
    project_info['batch_size'] = 100
    project_info['optim'] = 'Adadelta'
    project_info['train_loss'] = losses
    project_info['train_accu'] = accu_t
    project_info['valid_accu'] = accues
    project_info['preprocess'] = "transforms.ToTensor(),transforms.CenterCrop(24),"
    if not os.path.exists(ROOTDIR):os.makedirs(ROOTDIR)
    with open(f'{ROOTDIR}/result.json','w') as f:
        json.dump(project_info,f)
    torch.save(model.state_dict(),f"{ROOTDIR}/weight.pt")
