import os
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from function import speed_benchmark_pure
from utils import *
import torch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-D","--virtual_bond", default=6  , type=int,help="virtual bond dimension")
parser.add_argument("-I","--iters"       , default=15 , type=int,help="repeat times")
parser.add_argument("-N","--num"         , default=20 , type=int,help="MPS length")
parser.add_argument("-P","--physics_bond", default=3 ,  type=int,help="physics bond dimension")
parser.add_argument("-B","--B_list"      , default="1,2,3, 5,10,20,50,100,200,400,1000", type=str,help="batch size list")
parser.add_argument("-o","--output_dir" , default="./result" , type=str,help="output_dir")
args = parser.parse_args()

iters= args.iters
bd   = args.virtual_bond
pd   = args.physics_bond
cd   = 10
num  = args.num
batch_num_list=[int(t) for t in args.B_list.split(',')]
performance_table_path = os.path.join(args.output_dir,f"GgTNFrameWork.bd{bd}pd{pd}cd{cd}num{num}_pure.json")

base_func_list= [way_for_loop_einsum,
                 #way_batch_tensor_network_2,
                 #way_batch_tensor_network_pure,
                 #way_for_efficient_coding
                 ]
#base_func_list= [way_batch_tensor_network_pure]
task_list =[]
for function in base_func_list:
    for backend in [#'numpy',
                    #'pytorch',
                    #'torchgpu',
                    'tensorflow'
                    ]:
        task_list.append([backend,function])
# make sure do tensorflow last so there is no GPU memory problem
# task_list.append(['tensorflow',way_vectorized_map_einsum])

#task_list = [['pytorch',way_batch_tensor_network_pure],['torchgpu',way_batch_tensor_network_pure]]
if not os.path.exists(performance_table_path):
    performance_table={}
else:
    with open(performance_table_path,"r") as f:
        performance_table= json.load(f)
for backend,function in task_list:
    try:
        print(f"{backend}+{function.__name__}")
        if backend not in performance_table:performance_table[backend]={}
        if function.__name__ not in performance_table[backend]:
            performance_table[backend][function.__name__]={}

        result = speed_benchmark_pure(function,backend=backend,
                        batch_num_list=batch_num_list,iters=iters,bd = bd,pd = pd,cd = cd,num=num)
        for batch,cost in result.items():
            performance_table[backend][function.__name__][batch]=cost
        with open(performance_table_path,"w") as f:
            json.dump(performance_table,f)
    except:
        continue
    torch.cuda.empty_cache()
with open(performance_table_path,"w") as f:
    json.dump(performance_table,f)
