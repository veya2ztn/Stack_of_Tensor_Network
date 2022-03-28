import numpy as np
import torch
import tensornetwork as tn
import opt_einsum as oe
import os,json
from tensornetwork.network_components import get_all_nondangling,get_all_dangling
from tensornetwork.contractors.opt_einsum_paths.utils import get_subgraph_dangling,get_all_edges

def get_sublist_from_node_list(node_list,outlist = None):
    for edge in get_all_edges(node_list):
        if edge.name == '__unnamed_edge__':
            if edge.node1 is not None and edge.node2 is not None:
                edge.name= f'{edge.node1.name}:{edge.axis1}<->{edge.node2.name}:{edge.axis2}'
            else:
                edge.name= f"{edge.node1}:{edge.axis1}"
    class edges_name_mapper:
        name_to_idx = {}
        def get_index(self,name):
            if name not in self.name_to_idx:
                self.name_to_idx[name]=len(self.name_to_idx)
            return self.name_to_idx[name]
    mapper = edges_name_mapper()

    sublist_list = [[mapper.get_index(e.name)for e in t.edges] for t in node_list]

    if outlist is None:
        outlist = [mapper.get_index(e.name) for e in get_all_dangling(node_list)]
    else:
        outlist = [mapper.get_index(e.name) for e in outlist]
    return node_list,sublist_list,outlist

def create_templete_2DTN_tn(tn2D_shape_list,engine = np.random.randn):
    if engine is not np.random.randn:
        tn.set_default_backend("pytorch")
    else:
        tn.set_default_backend("numpy")
    node_array      = []
    W = len(tn2D_shape_list)
    H = len(tn2D_shape_list[0])
    for i in range(W):
        node_line = []
        for j in range(H):
            node = tn.Node(engine(*tn2D_shape_list[i][j]),name=f"{i}-{j}")
            node_line.append(node)
        node_array.append(node_line)
    #row
    for i in range(W-1):
        for j in range(H-1):
            tn.connect(node_array[i][j][1],node_array[i][j+1][-1],name=f'{i},{j}|{i},{j+1}')
    #last row
    i=W-1
    for j in range(H-1):
            tn.connect(node_array[i][j][0],node_array[i][j+1][-1],name=f'{i},{j}|{i},{j+1}')
    #col
    for j in range(H-1):
        for i in range(W-2):
            tn.connect(node_array[i][j][0],node_array[i+1][j][2],name=f'{i},{j}|{i+1},{j}')
        i=W-2
        tn.connect(node_array[i][j][0],node_array[i+1][j][1],name=f'{i},{j}|{i+1},{j}')
    j = H-1
    for i in range(W-2):
        tn.connect(node_array[i][j][0],node_array[i+1][j][1],name=f'{i},{j}|{i+1},{j}')
    i=W-2
    tn.connect(node_array[i][j][0],node_array[i+1][j][0],name=f'{i},{j}|{i+1},{j}')
    node_list = [item for sublist in node_array for item in sublist]
    node_list,sublist_list,outlist = get_sublist_from_node_list(node_list)
    return node_list,sublist_list,outlist

def get_optim_path_by_oe_from_tn(node_list,optimize='random-greedy-128'):
    operands = []
    for node in node_list:
        operands+=[node.tensor,[edge.name for edge in node.edges]]
    operands+= [[edge.name for edge in get_all_dangling(node_list)]]
    path,info = oe.contract_path(*operands,optimize=optimize)
    return path,info
def sub_network_tn(tn2D_shape_list):
    node_array      = []
    W = len(tn2D_shape_list)
    H = len(tn2D_shape_list[0])
    for i in range(W):
        node_line = []
        for j in range(H):
            node = tn.Node(np.random.randn(*tn2D_shape_list[i][j]),name=f"{i}-{j}")
            node_line.append(node)
        node_array.append(node_line)
    #row
    for i in range(W):
        for j in range(H-1):
            tn.connect(node_array[i][j][1],node_array[i][j+1][-1],name=f'{i},{j}|{i},{j+1}')
    for j in range(H):
        for i in range(W-1):
            tn.connect(node_array[i][j][0],node_array[i+1][j][2],name=f'{i},{j}|{i+1},{j}')
    node_list = [item for sublist in node_array for item in sublist]
    for edge in get_all_edges(node_list):
        if edge.name == '__unnamed_edge__':
            if edge.node1 is not None and edge.node2 is not None:
                edge.name= f'{edge.node1.name}:{edge.axis1}<->{edge.node2.name}:{edge.axis2}'
            else:
                edge.name= f"{edge.node1}:{edge.axis1}"
    class edges_name_mapper:
        name_to_idx = {}
        def get_index(self,name):
            if name not in self.name_to_idx:
                self.name_to_idx[name]=len(self.name_to_idx)
            return self.name_to_idx[name]
    mapper = edges_name_mapper()
    sublist_list = [[mapper.get_index(e.name)for e in t.edges] for t in node_list]
    outlist = [mapper.get_index(e.name) for e in get_all_dangling(node_list)]
    return node_list,sublist_list,outlist

def read_path_from_offline(shape_array):
    with open("models/arbitrary_shape_path_recorder.json",'r') as f:
        path_pool = json.load(f)
    shape_array_string = convert_array_shape_to_string(shape_array)
    print(shape_array_string)
    if shape_array_string not in path_pool:
        return None
    else:
        return path_pool[shape_array_string]
def convert_array_shape_to_string(array_shape):
    '''
    convert array shape, for example
                [[ (D,D) , (D,D,D) , (D,D,D) , (D,D) ],
                 [(D,D,D),(D,D,D,D),(D,D,D,D),(D,D,D)],
                 [(D,D,D),(D,D,D,D),(D,D,D,D),(D,D,D)],
                 [ (D,D) , (D,D,D) , (D,D,D) , (D,D) ]]
    '''
    return "|".join([",".join([str(t) for t in line]) for line in array_shape])

def get_best_path(tn2D_shape_list,store=None,type='sub'):
    saveQ = False
    path_store_path = None
    if store is None:
        node_list,sublist_list,outlist = sub_network_tn(tn2D_shape_list) if type =='sub' else create_templete_2DTN_tn(tn2D_shape_list)
        path,info                      = get_optim_path_by_oe_from_tn(node_list)
    else:
        array_string=convert_array_shape_to_string(tn2D_shape_list)
        if isinstance(store,str):
            path_store_path = store
            if os.path.exists(path_store_path):
                with open(path_store_path,'r') as f:store = json.load(f)
            else:
                store={}
        assert isinstance(store,dict)
        if array_string not in store:
            node_list,sublist_list,outlist = sub_network_tn(tn2D_shape_list) if type =='sub' else create_templete_2DTN_tn(tn2D_shape_list)
            path,info                      = get_optim_path_by_oe_from_tn(node_list)
            store[array_string]={}
            store[array_string]['path']=path
            store[array_string]['outlist']=outlist
            store[array_string]['sublist_list']=sublist_list
            saveQ = True
        path        = store[array_string]['path']
        sublist_list= store[array_string]['sublist_list']
        outlist     = store[array_string]['outlist']
        if saveQ and path_store_path is not None:
            with open(path_store_path,'w') as f:
                json.dump(store,f)
    return path,sublist_list,outlist


def full_size_array_string(*operands):
    if isinstance(operands[0],str):
        equation = operands[0]
        tensor_l = operands[1:]
        array_string=equation+"?"+",".join([str(tuple(t.shape)) for t in tensor_l])
    else:
        array_string=[]
        for t in operands:
            if isinstance(t,list):
                # t is sublist
                shape =[ l for l in t if l is not Ellipsis]
                shape = "["+",".join([str(t) for t in shape])+"]"
            else:
                # t is a tensor
                shape = list(t.shape)
                shape = "("+",".join([str(t) for t in shape])+")"
            array_string.append(shape)
        array_string=','.join(array_string)
    return array_string
def get_best_path_via_oe(equation,tensor_l,store=None):
    saveQ = False
    path_store_path = None
    if store is None:
        path = oe.contract_path(equation, *tensor_l)[0]
    else:
        array_string=equation+"?"+",".join([str(tuple(t.shape)) for t in tensor_l])
        if isinstance(store,str):
            path_store_path = store
            if os.path.exists(path_store_path):
                with open(path_store_path,'r') as f:store = json.load(f)
            else:
                store={}
        assert isinstance(store,dict)
        if array_string not in store:
            path = oe.contract_path(equation, *tensor_l)[0]
            store[array_string]={}
            store[array_string]['path']=path
            saveQ = True
        path        = store[array_string]['path']
        if saveQ and path_store_path is not None:
            with open(path_store_path,'w') as f:
                json.dump(store,f)
    return path
def get_chain_contraction(tensor):
    size   = int(tensor.shape[0])
    while size > 1:
        half_size = size // 2
        nice_size = 2 * half_size
        leftover  = tensor[nice_size:]
        tensor    = torch.einsum("mbik,mbkj->mbij",tensor[0:nice_size:2], tensor[1:nice_size:2])
        #(k/2,NB,D,D),(k/2,NB,D,D) <-> (k/2,NB,D,D)
        tensor   = torch.cat([tensor, leftover], axis=0)
        size     = half_size + int(size % 2 == 1)
    return tensor.squeeze(0)
def batch_contract_mps_mpo(mps_list,mpo_list):
    # mps_list                                    --D--|--D--
    # (D,D)-(D,D,D)-(D,D,D)-...-(D,D,D)-(D,D)          D
    #  -b-    -c-  -b-   -c-  -b-     -b-
    # |a         |a         |a          |a
    # the order i.e. 'abcd' counterclockwise and start from the down index. (so down index must a )
    #  mpo_list
    # (P,D,P)-(D,P,D,P)-(D,P,D,P)-...-(D,P,D,P)-(D,P,P)
    #  |c            |c        |c           |b
    #   -b-      -d-  -b-  -d-  -b-     -c-
    # |a            |a        |a           |a
    assert len(mps_list) > 2
    assert len(mpo_list) > 2
    stack_unit_mps = (len(mps_list)==3 and len(mps_list[0].shape)+ 2 == len(mps_list[1].shape))
    stack_unit_mpo = (len(mpo_list)==3 and len(mpo_list[0].shape)+ 2 == len(mpo_list[1].shape))
    mps_left,mps_rigt = mps_list[0],mps_list[-1]
    mpo_left,mpo_rigt = mpo_list[0],mpo_list[-1]
    new_mps_list= []
    tensor = torch.einsum("  kab,kcda->kcbd",mps_left,mpo_left).flatten(-2,-1)
    new_mps_list.append(tensor)
    if stack_unit_mps and stack_unit_mpo:
        mps_inne = mps_list[1]
        mpo_inne = mpo_list[1]
        tensor = torch.einsum("lkabc,lkdeaf->lkdbecf",mps_inne,mpo_inne).flatten(-4,-3).flatten(-2,-1)
        new_mps_list.append(tensor)
    else:
        mps_inne = mps_list[1:-1] if not stack_unit_mps else list(*mps_list[1])
        mpo_inne = mpo_list[1:-1] if not stack_unit_mpo else list(*mpo_list[1])
        for mps,mpo in zip(mps_inne,mpo_inne):
            tensor =torch.einsum("kabc,kdeaf->kdebcf",mps,mpo).flatten(-4,-3).flatten(-2,-1)
            new_mps_list.append(tensor)
    if len(mpo_rigt.shape)==4:
        tensor = torch.einsum("  kab,kcad->kcbd",mps_rigt,mpo_rigt).flatten(-2,-1)
    elif len(mpo_rigt.shape)==5:
        tensor = torch.einsum("  kab,kocad->kocbd",mps_rigt,mpo_rigt).flatten(-2,-1)
    else:raise NotImplementedError
    new_mps_list.append(tensor)
    return new_mps_list
def batch_contract_mps_mps(mps_list,mpo_list):
    # mps_list                                    --D--|--D--
    # (D,D)-(D,D,D)-(D,D,D)-...-(D,D,D)-(D,D)          D
    # (D,D)-(D,D,D)-(D,D,D)-...-(D,D,D)-(D,D,O)
    #  -b-    -c-  -b-   -c-  -b-     -b-
    # |a         |a         |a          |a
    #
    # |b            |b        |b        |b
    #  -a-      -c-  -a-  -c-  -a-   -c- -a
    assert len(mps_list) > 2
    assert len(mpo_list) > 2
    stack_unit_mps = (len(mps_list)==3 and len(mps_list[0].shape)+ 2 == len(mps_list[1].shape))
    stack_unit_mpo = (len(mpo_list)==3 and len(mpo_list[0].shape)+ 2 == len(mpo_list[1].shape))
    mps_left,mps_rigt = mps_list[0],mps_list[-1]
    mpo_left,mpo_rigt = mpo_list[0],mpo_list[-1]
    new_mps_list= []
    tensor = torch.einsum("  kab,kca->kbc",mps_left,mpo_left).flatten(-2,-1)
    new_mps_list.append(tensor)
    if stack_unit_mps and stack_unit_mpo:
        mps_inne = mps_list[1]
        mpo_inne = mpo_list[1]
        tensor = torch.einsum("lkabc,lkdae->lkbdce",mps_inne,mpo_inne).flatten(-4,-3).flatten(-2,-1)
        new_mps_list.append(tensor)
    else:
        mps_inne = mps_list[1:-1] if not stack_unit_mps else list(*mps_list[1])
        mpo_inne = mpo_list[1:-1] if not stack_unit_mpo else list(*mpo_list[1])
        for mps,mpo in zip(mps_inne,mpo_inne):
            tensor =torch.einsum("kabc,kdae->kbdce",mps,mpo).flatten(-4,-3).flatten(-2,-1)
            new_mps_list.append(tensor)
    tensor = torch.einsum("  kab,okac->kobc",mps_rigt,mpo_rigt).flatten(-2,-1)

    new_mps_list.append(tensor)
    return new_mps_list
def structure_operands(tensor_list,sublist_list,outlist,type='torch'):
    type='torch'
    if type == "torch":
        operands=[]
        for tensor,sublist in zip(tensor_list,sublist_list):
            operand = [tensor,[...,*sublist]]
            operands+=operand
        operands+= [[...,*outlist]]
    elif type == "oe":
        operands=[]
        for tensor,sublist in zip(tensor_list,sublist_list):
            operand = [tensor,[*sublist]]
            operands+=operand
        operands+= [[*outlist]]
    return operands
