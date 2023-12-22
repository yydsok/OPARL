import subprocess
from termcolor import colored
from tqdm import tqdm
import torch
import argparse


def run_commands_in_parallel(commands):
    processes = [subprocess.Popen(cmd, shell=True) for cmd  in commands]
    for p in processes:
        p.wait()

def split_list(input_list, K):
    return [input_list[i:i+K] for i in range(0, len(input_list), K)]



if __name__ == '__main__':
    # with graph pool
    torch.cuda.empty_cache()

    num_cmds = 12
    cmd_biao = '/root/miniconda3/bin/python /root/yyds/main.py'
    cmds =[]
    seeds = [1234,5678,9123,4567,8912,3456,7891,2345,6789,2018,2024,2080]
    for seed in seeds:
        cmd = cmd_biao+"  --seed {} ".format(seed)
        cmds.append(cmd)
          
    commands_batches = split_list(cmds,num_cmds)

    cnt = 0
    for commands in tqdm(commands_batches):
        cnt +=1
        print (colored(f"--------current progress: {cnt} out of {len(commands_batches)}---------", 'blue','on_white'))
        run_commands_in_parallel(commands)