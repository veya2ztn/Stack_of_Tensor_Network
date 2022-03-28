import numpy as np
import tensorflow as tf
import torch
import time
import tensornetwork as tn
from fastprogress.fastprogress import master_bar, progress_bar
from utils import get_BatchedTensorNetwork
generator_for_backend={'numpy':['numpy',np.random.rand],
                 'tensorflow':['tensorflow',lambda *a:tf.random.normal(a)],
                    'pytorch':['pytorch',torch.rand],
                   'torchgpu':['pytorch',lambda *a:torch.rand(a).to('cuda:0')]
                   }
def speed_benchmark(function,batch_num_list=[1,5,10,50,100],iters=20,bd = 3,pd = 2,cd = 10,num=21,backend='numpy'):
    bkend,generator = generator_for_backend[backend]
    tn.set_default_backend(bkend)
    core_tensornetwork = ([generator(bd,pd)]+
                          [generator(bd,cd,bd)]+
                          [generator(bd,pd,bd)]*(num-3)+
                          [generator(bd,pd)])
    hidden_node_idx = [i for i,v in enumerate(core_tensornetwork) if v.shape!=(bd,cd,bd)]
    input_data      = [generator(batch_num,num-1,pd) for batch_num in batch_num_list]
    mb              = master_bar(range(len(batch_num_list)))
    performance     = {}
    if function.__name__=="way_batch_tensor_network_pure":
        input_data  = [get_BatchedTensorNetwork(inp,bkend) for inp in input_data]
    for i in mb:
        input     = input_data[i]
        batch_num = len(input) if function.__name__!="way_batch_tensor_network_pure" else len(input[0])
        mb.main_bar.comment = f'{backend}+{function.__name__}+{batch_num}'

        t         = time.process_time()
        for j in progress_bar(range(iters), parent=mb):
            with torch.no_grad():
                result = function(core_tensornetwork,input,hidden_node_idx)
        elapsed_time = time.process_time() - t
        cost   = elapsed_time/iters

        performance[batch_num]=cost
    return performance
def speed_benchmark_pure(function,batch_num_list=[1,5,10,50,100],iters=20,bd = 3,pd = 2,cd = 10,num=21,backend='numpy'):
    bkend,generator = generator_for_backend[backend]
    tn.set_default_backend(bkend)
    core_tensornetwork = ([generator(bd,pd)]+
                          [generator(bd,pd,bd)]*(num-2)+
                          [generator(bd,pd,cd)])

    input_data      = [generator(batch_num,num,pd) for batch_num in batch_num_list]
    mb              = master_bar(range(len(batch_num_list)))
    performance     = {}
    if 'pure' in function.__name__:
        input_data  = [get_BatchedTensorNetwork(inp,bkend) for inp in input_data]


    for i in mb:
        input     = input_data[i]
        batch_num = len(input) if 'pure' not in function.__name__ else len(input[0])
        mb.main_bar.comment = f'{backend}+{function.__name__}+{batch_num}'

        if 'batch_tensor_network' in function.__name__:
            with torch.no_grad():
                contraction_info = function(core_tensornetwork,input)
        else:
            contraction_info = None
        t         = time.process_time()
        for j in progress_bar(range(iters), parent=mb):
            with torch.no_grad():
                if contraction_info is not None:
                    result = function(core_tensornetwork,input,contraction_info=contraction_info)
                else:
                    result = function(core_tensornetwork,input)
        elapsed_time = time.process_time() - t
        cost   = elapsed_time/iters

        performance[batch_num]=cost
    return performance
