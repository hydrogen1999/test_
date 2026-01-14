# Setup env
Run:
```
conda env create -f <env_name>.yml
```
<env_name>: can be 'spec-ising' or 'spec-pt2'

# Adjust configs and code
All configs are saved in ./configs/

num_seeds: the number of seeds, normally is 1, but 10 for spectral annealing (SpecAnn)
For example:
```
SB_50:
    solver_name: "SB"
    params:
        agents: 50
```
SB_50: the made-up name for each experiment/test
solver_name: the core solver that the experiment/test based on
params: set of params adjusted according to the purpose of the experiment/test

# Generate job files
generate_cluster_jobs.py: generate job files for CPU runs
generate_cluster_jobsGPU.py: generate job files for GPU runs

In generate_cluster_jobs.py,
SOLVERS: contains the list of made-up solver names defined previously in the config file
DATASETS: contains the list of datasets for the experiment, typically should include only 1 dataset for safety purpose or in case GPU
SLURM_CONFIG: contains setup for each job file, should be untouched for GPU; in case of CPU, the 'memory' can be changed to optimized the cluster's performance, should be based on some benchmark previously
INSTANCE_MEMORY: SHOULD BE UNTOUCHED

Command to run:
```
module purge
python generate_cluster_jobs.py
```
In case of parallel tempering solver (PT), the command changes to
```
python generate_cluster_jobs.py --conda-env spec-pt2
```

The output job files are saved to ./cluster_jobs/
The list of job file names is saved to job_list_<env_name>.txt. The order of job file names is also the order of jobs in slurm queue. Optimize this order to optimize the cluster's performance!
Change the file name to job_list.txt (to make sure everything is carefully checked)

# Submit jobs
Run:
```
submit_limited.sh
```
All job files after submited will be moved to folder ./cluster_jobs/done
All job status will be shown in squeue immediately.
During the run, check the output and log files in ./cluster_logs periodically to prevent any possible error/abnormality.
All results will be saved to ./results/<dataset_name>/<made-up_solver_name>/

# Prepare for the next runs
Delete/backup job_list.txt
Make sure there is no duplicated made-up solver names (to prevent overwriting)
