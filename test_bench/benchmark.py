import time
from balsam_setup import balsam_initialize
from balsam_workflow import run_balsam
from parsl_workflow import run_parsl
from parsl_setup import set_config
import parsl

def initialize(version):
    if version == "balsam":
        print('Do balsam initialization')
        balsam_initialize()
    elif version == 'parsl':
        print("No init")
        # config = set_config()
        # parsl.load(config)
        # print('Loaded parsl config')
        # print(config)
    else:
        raise("Unknown benchmark version")

def run(version, workflow_type):

    if version == "balsam":
        run_balsam()
    elif version == "parsl":
        workflow = run_parsl()
        #print(workflow.result())
        print([t.result() for t in workflow])
    else:
        raise("Unknown benchmark version")
    return

if __name__ == '__main__':

    version = "parsl"
    workflow_type = "uniform_mini_sleep.json"

    t0 = time.perf_counter()
    
    initialize(version=version)
    run(version=version, 
       workflow_type=workflow_type)

    total_run_time = time.perf_counter() - t0
    print(f"*** FOM: Total run time: {total_run_time} seconds ***")