import os
import json
from function import memory_benchmark
from utils import *
import torch

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-D","--D_list", default="50", type=str,help="virtual_bond list")
parser.add_argument("-B","--B_list", default="10,50,100,200,500,800,1000,1500", type=str,help="batch size list")
parser.add_argument("-P","--physics_bond", default=3 ,  type=int,help="physics bond dimension")
parser.add_argument("-L","--L_list", default="21", type=str,help="size list")
parser.add_argument("-f","--function_list" , default="EC+BTN" , type=str,help="function_list")
parser.add_argument("-o","--output_dir" , default="./result" , type=str,help="output_dir")

args = parser.parse_args()

D_list = [int(t) for t in args.D_list.split(',')]
L_list = [int(t) for t in args.L_list.split(',')]
B_list = [int(t) for t in args.B_list.split(',')]

y,x = np.meshgrid(D_list,L_list)
y=y.flatten()
x=x.flatten()
LD_List = [(a,b) for a,b in zip(x,y)]

pd   = args.physics_bond
cd   = 10
batch_num_list=B_list


base_func_list= [#way_for_loop_einsum,
                 way_for_efficient_coding,
                 way_batch_tensor_network_2,

                 ]

performance_table_path = os.path.join(args.output_dir,f"GgTNFrameWork.Memory.Increase_batch.json")
backend = 'torchgpu'
performance_table={}
# if not os.path.exists(performance_table_path):
#     performance_table={}
# else:
#     with open(performance_table_path,"r") as f:
#         performance_table= json.load(f)
from tqdm import tqdm
for num,bd in tqdm(LD_List):
    num = int(num)
    bd  = int(bd)
    for function in base_func_list:
        fname= str(function.__name__)
        print(f"{fname}+N{num}B{bd}P{pd}")
        if fname not in performance_table:performance_table[fname]={}
        if  num  not in performance_table[fname]:performance_table[fname][num]={}
        if  bd   not in performance_table[fname][num]:
            result = memory_benchmark(function,batch_num_list=batch_num_list,bd = bd,pd = pd,cd = cd,num=num)
            performance_table[fname][num][bd]=result
            with open(performance_table_path,"w") as f:
                json.dump(performance_table,f)
        for batch,cost in performance_table[fname][num][bd].items():
            print(f"batch:{batch} -> cost:{cost}")
    # except:
    #     continue
with open(performance_table_path,"w") as f:
    json.dump(performance_table,f)
