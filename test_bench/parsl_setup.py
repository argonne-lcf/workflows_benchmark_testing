from parsl import python_app
from parsl.config import Config

# PBSPro is the right provider for Polaris:
from parsl.providers import PBSProProvider, LocalProvider
# The high throughput executor is for scaling to HPC systems:
from parsl.executors import HighThroughputExecutor
# You can use the MPI launcher, but may want the Gnu Parallel launcher, see below
from parsl.launchers import MpiExecLauncher, GnuParallelLauncher
# address_by_interface is needed for the HighThroughputExecutor:
from parsl.addresses import address_by_interface

def set_config(queue="debug", 
               project="datascience", 
               gpus_per_task=1, 
               cpu_threads_per_task=8,
               ranks_per_task=1,
               provider="Local",
               machine_config={"gpus_per_node":4, "threads_per_node": 64}, # Default is Polaris
               ):
    if provider == "PBS":
        provider=PBSProProvider(
                        launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"),
                        # Which launcher to use?  Check out the note below for some details.  Try MPI first!
                        # launcher=GnuParallelLauncher(),
                        account=project,
                        queue=queue,
                        select_options="ngpus=4",
                        # PBS directives (header lines): for array jobs pass '-J' option
                        scheduler_options="#PBS -l filesystems=home:eagle:grand",
                        # Command to be run before starting a worker, such as:
                        worker_init="module use /soft/modulefiles; module load conda; conda activate balsam; export PYTHONPATH=/eagle/datascience/csimpson/alcf4/workflows_benchmark_testing/test_bench:$PYTHONPATH",
                        # number of compute nodes allocated for each block
                        nodes_per_block=1,
                        init_blocks=1,
                        min_blocks=0,
                        max_blocks=1, # Can increase more to have more parallel jobs
                        cpus_per_node=64,
                        walltime="1:00:00"
                    )
    else:
        provider=LocalProvider(
                        launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"),
                        worker_init=f"module use /soft/modulefiles; module load conda; conda activate balsam; export PYTHONPATH=$(pwd):$PYTHONPATH",
                        # number of compute nodes allocated for each block
                        nodes_per_block=1,
                        init_blocks=1,
                        min_blocks=0,
                        max_blocks=1, # Can increase more to have more parallel jobs
                    )
    config = Config(
            executors=[
                HighThroughputExecutor(
                    label="htex",
                    heartbeat_period=15,
                    heartbeat_threshold=120,
                    available_accelerators=machine_config["gpus_per_node"]//gpus_per_task, # if this is set, it will override other settings for max_workers if set
                    max_workers=machine_config["gpus_per_node"]//gpus_per_task,
                    cores_per_worker=cpu_threads_per_task,
                    address=address_by_interface("bond0"),
                    cpu_affinity="alternating",
                    prefetch_capacity=0,
                    provider=provider,
                ),
            ],
            retries=2,
            app_cache=True,
    )
    return config

@python_app
def parsl_sleeper(app_path, args):
    import sys
    sys.path.append(app_path)
    from apps import sleeper
    return sleeper(**args)