# FAIR-MAST Data Preprocessing

Data preprocessing of the FAIR-MAST data for foundation modelling.

Initialised as part of HNCDI project Fusion Plasma Modelling HT07632.

For usage on [CSD3](https://docs.hpc.cam.ac.uk/hpc/index.html).

## Initial setup

1. Connect to CSD3 following the instructions in the [user guide](https://docs.hpc.cam.ac.uk/hpc/user-guide/quickstart.html).
2. Install Miniforge by following the instructions [here](https://github.com/conda-forge/miniforge). In brief:  
   1. Find the latest [release](https://github.com/conda-forge/miniforge/releases/).
   2. Download the installer. For example:
      ```bash
      cd ~
      wget https://github.com/conda-forge/miniforge/releases/download/24.11.3-2/Miniforge3-24.11.3-2-Linux-x86_64.sh
      ```
   3. Run the installer. For example:
      ```bash
      chmod +x Miniforge3-24.11.3-2-Linux-x86_64.sh
      ./Miniforge3-24.11.3-2-Linux-x86_64.sh
      ```
   4. Source the base environment:
      ```bash
      source ~/miniforge3/bin/activate
      ```
3. Set up the project virtual environment:
   1. Move to the directory containing this repository. For example:
      ```bash
      cd ~/hncdi-fusion-plasma/fairmast-data-preprocessing
      ```
   2. Use conda to set up the virtual environment and install the dependencies:
      ```bash
      conda env create -f environment.yml
      ```
4. Activate the project conda environment:
   ```bash
   conda activate fairmast-data-preprocessing
   ```

## CSD3 instructions

Running computational workloads on CSD3 (including Jupyter Notebooks) should be done using the worker nodes.

Start interactive job with 32 CPUs per task, 1 Node for 1 hour:

```bash
sintr -A ukaea-ap002-cpu -p ukaea-icl -N1 --ntasks=1 --cpus-per-task=32 -t 1:00:00
```

Start interactive job with 4 GPUs, 1 Node, 1 task (n) for 2 hours:
```bash
sintr --gres=gpu:4 -A ukaea-ap002-gpu -p ukaea-amp -N1 -n1 -t 2:0:0
```

Check accounts and partitions that I am allowed to use, (example for my acount ir-lore2):
```bash
sacctmgr show associations user=$USER format=Cluster,User,Account,Partition


 Cluster       User    Account  Partition 
---------- ---------- ---------- ---------- 
      csd3   ir-lore2 ukaea-ap0+    desktop 
      csd3   ir-lore2 ukaea-ap0+ ukaea-icl+ 
      csd3   ir-lore2 ukaea-ap0+  ukaea-icl 
      csd3   ir-lore2 ukaea-ap0+ ukaea-spr+ 
      csd3   ir-lore2 ukaea-ap0+ ukaea-spr+ 
      csd3   ir-lore2 ukaea-ap0+  ukaea-spr 
      csd3   ir-lore2 ukaea-ap0+    desktop 
      csd3   ir-lore2 ukaea-ap0+ ukaea-icl+ 
      csd3   ir-lore2 ukaea-ap0+  ukaea-icl 
      csd3   ir-lore2 ukaea-ap0+ ukaea-spr+ 
      csd3   ir-lore2 ukaea-ap0+ ukaea-spr+ 
      csd3   ir-lore2 ukaea-ap0+  ukaea-spr 
      csd3   ir-lore2 ukaea-ap0+  ukaea-amp 
```

Show partitions, example for ukaea-icl:
```bash
scontrol show partition ukaea-amp
```

```
AllowGroups=ALL AllowAccounts=ALL AllowQos=sys0,sys1,support-gpu,covid0,intr,ukaea-gpu1,gpu0,gpul,gpu1,gpu2,gpu3,dirac-gpu1,dirac-gpu3
AllocNodes=ALL Default=NO QoS=N/A
DefaultTime=00:10:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
MaxNodes=UNLIMITED MaxTime=7-00:00:00 MinNodes=0 LLN=NO MaxCPUsPerNode=UNLIMITED MaxCPUsPerSocket=UNLIMITED
Nodes=gpu-q-[48-60,64,73-80]
PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=NO
OverTimeLimit=NONE PreemptMode=OFF
State=UP TotalCPUs=2816 TotalNodes=22 SelectTypeParameters=NONE
JobDefaults=(null)
DefMemPerCPU=8000 MaxMemPerCPU=8000
TRES=cpu=2816,mem=22000G,node=22,billing=2816,gres/gpu=88
ResumeTimeout=GLOBAL SuspendTimeout=GLOBAL SuspendTime=1800 PowerDownOnIdle=NO
```

Show nodes list of a specific partition, for instance 
```
sinfo -p ukaea-amp -N
```

```
NODELIST   NODES PARTITION STATE 
gpu-q-48       1 ukaea-amp idle~ 
gpu-q-49       1 ukaea-amp alloc 
gpu-q-50       1 ukaea-amp alloc 
gpu-q-51       1 ukaea-amp alloc 
gpu-q-52       1 ukaea-amp alloc 
gpu-q-53       1 ukaea-amp alloc 
gpu-q-54       1 ukaea-amp alloc 
gpu-q-55       1 ukaea-amp alloc 
gpu-q-56       1 ukaea-amp alloc 
gpu-q-57       1 ukaea-amp alloc 
gpu-q-58       1 ukaea-amp alloc 
gpu-q-59       1 ukaea-amp idle~ 
gpu-q-60       1 ukaea-amp idle~ 
gpu-q-64       1 ukaea-amp idle~ 
gpu-q-73       1 ukaea-amp alloc 
gpu-q-74       1 ukaea-amp alloc 
gpu-q-75       1 ukaea-amp alloc 
gpu-q-76       1 ukaea-amp alloc 
gpu-q-77       1 ukaea-amp alloc 
gpu-q-78       1 ukaea-amp alloc 
gpu-q-79       1 ukaea-amp alloc 
gpu-q-80       1 ukaea-amp alloc 
```
    

To run a Jupyter notebook on a worker node using VSCode:

1. On the local machine, add an entry to `~/.ssh/config`:
   ```
   Host csd3
     User <user-name>
     HostName login-cpu.hpc.cam.ac.uk
     IdentityFile </path/to/id_rsa>
   ```
2. In VSCode, install the "Remote Development" extension pack (if not already installed).
3. In VSCode, run the command "Remote-SSH: Connect to Host" and select "csd3" from the dropdown menu. VSCode will now connect to a VSCode server running on a CSD3 login node.
4. In the VSCode terminal, start an interactive session using the command above. Note down the node address, which appears as part of the terminal prompt prefix with the format `[<user-name>@<node-address>]` and will be something like `cpu-q-123`.
5. In the interactive session, start a Jupyter notebook server:
   ```bash
   jupyter notebook --no-browser --ip=* --port=8081
   ```
   Wait for the server to start then copy the address, which should have the format:
   ```
   http://localhost:8081/tree?token=<token>
   ```
6. In another VSCode terminal (i.e., not the interactive one), tunnel to the worker node and forward the Jupyter server port:
   ```bash
   ssh -L 8081:localhost:8081 <user-name>@<node-address>
   ```
7. In VSCode, open up a `.ipynb` file and connect to the server running on the worker node:
   1. Run the command `Notebook: Select Notebook Kernel`.
   2. Click "Select Another Kernel".
   3. Select "Existing Jupyter Server".
   4. Paste in the server address noted down before and hit enter. Hit enter on the subsequent input boxes to accept the default values, or modify them if needed.

Any cells executed in the notebook will now execute on the kernel running on the reserved cores of the worker node.