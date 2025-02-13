#!/usr/bin/env python
""" Copyright (c) CERFACS (all rights reserved)
@file       placement4mpmd.py
@details    MPI placement generator
@autor      A. Serhani
@email      phydll@cerfacs.fr

Generate a MPI configuration files to carry out processes placement for multi-node PhyDLL

Inputs:
    --Run       (str)   Runmode: IntelMPI mpirun, OpenMPI mpirun, srun
    --NpPhy     (int)   Number of Physical solver processes
    --NpDL      (int)   Number of Python process
    --PHYEXE    (str)   Physical solver executable
    --DLEXE     (str)   DL executable

Output:
    machinefile_*-*-*       (txt file)  machinefile/hostfile (for intel mpirun)
    rankfile_*-*-*          (txt file)  rankfile (for openmpi mpirun)
    phydll_mpmd_*-*-*.conf  (txt file)  multi-prog file (for srun)

Example:
    python placement4mpmd.py -r ompi --NpPHY 64 --NpDL 8 --PHYEXE "PHYEXE" --DLEXE "python main.py"
"""
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--Run", default="impi", help="Run: impi, ompi, srun")
parser.add_argument("-nphy", "--NpPHY", type=int, default=28)
parser.add_argument("-ndl", "--NpDL", type=int, default=4)
parser.add_argument("--PHYEXE", type=str, default="PHYEXE")
parser.add_argument("--DLEXE", type=str, default="DLEXE")
parser.add_argument("--nPhyPerNode", type=int, default=8, help="Number of physical solver MPI-processes per node")
parser.add_argument("--nGPUsPerNode", type=int, default=4, help="Number of GPUs per node")
args = parser.parse_args()

cpu_total = args.NpPHY
gpu_total = args.NpDL
runmode = args.Run

def main():
    # #DEBUG
    # cpu_total = 15 # @!
    # gpu_total = 6 # @!
    # nodelist = ["node1\n", "node2\n", "node3\n"] # @!

    # Read nodelist from slurm
    os.system("scontrol show hostname $SLURM_JOB_NODELIST > ./nodelist.txt")
    nodelist_file = "./nodelist.txt"
    file = open(nodelist_file, "r")
    nodelist = file.readlines()
    file.close()
    os.system(f"rm {nodelist_file}")

    nnode = len(nodelist)
 
    # workaround to avoid SLURM hetjob --> only allocate GPU nodes and place DL processes on the last node in the allocation
    #cpu_pernode = cpu_total // nnode
    #gpu_pernode = gpu_total // nnode
    cpu_pernode = args.nPhyPerNode 
    gpu_pernode = args.nGPUsPerNode
    nnode_cpu = int(cpu_total / cpu_pernode)
    if gpu_total == 1:
        nnode_gpu = 1
    if gpu_total == 2:
        nnode_gpu = 1
    if gpu_total >= 4:
        nnode_gpu = int(gpu_total / gpu_pernode)
    print(f"PHYDLL Placement: number of cpu nodes: {nnode_cpu}")
    print(f"PHYDLL Placement: number of gpu nodes: {nnode_gpu}")
    ##########################################

    ntasks = cpu_total + gpu_total

    # Machinefile (mpirun impi)
    if runmode == "impi":
        mf_file = open(f"machinefile_{nnode}-{cpu_total}-{gpu_total}", "w")
        for i in range(nnode):
            mf_file.write(f"{nodelist[i][:-1]}:{cpu_pernode}\n")
        for i in range(nnode):
            mf_file.write(f"{nodelist[i][:-1]}:{gpu_pernode}\n")
        mf_file.close()

    # Rankfile (mpirun ompi)
    if runmode == "ompi":
        rk_file = open(f"rankfile_{nnode}-{cpu_total}-{gpu_total}", "w")
        for i in range(2 * nnode):
            if i < nnode:
                for j in range(cpu_pernode):
                    rk_file.write(f"rank {cpu_pernode*i+j}={nodelist[i][:-1]} slot={j//((cpu_pernode+1)//2)}:{j%((cpu_pernode+1)//2)}\n")
            elif i >= nnode:
                for j in range(gpu_pernode):
                    rk_file.write(f"rank {cpu_total-gpu_total+gpu_pernode*i+j}={nodelist[i-nnode][:-1]} slot={j//((gpu_pernode+1)//2)}:{j%((gpu_pernode+1)//2)}\n")
        rk_file.close()

    # Multi-prog conf (srun)
    if runmode == "srun":
        conf_file = open(f"./phydll_mpmd_{nnode}-{cpu_total}-{gpu_total}.conf", "w")
        conf_file.write(f"{0}-{cpu_total-1} {args.PHYEXE}\n")
        conf_file.write(f"{cpu_total}-{cpu_total+gpu_total-1} {args.DLEXE}")
        conf_file.close()

        machinefile = []
        #for i in range(nnode):
        for i in range(nnode_cpu):
            for j in range(cpu_pernode):
                machinefile.append(nodelist[i])
        #for i in range(nnode):
        for i in range(nnode_gpu):
            for j in range(gpu_pernode):
                machinefile.append(nodelist[nnode_cpu + i])

        mf_file = open(f"machinefile_{nnode}-{cpu_total}-{gpu_total}", "w")
        for i in range(ntasks):
            entry = machinefile[i][:-1]
            if i < ntasks - 1:
                entry = entry + ","
            mf_file.write(entry)
        mf_file.close()


if __name__ == "__main__":
    main()
