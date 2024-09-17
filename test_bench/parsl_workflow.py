import json
import time
import os
import parsl
from parsl_setup import parsl_sleeper, set_config
from parsl import python_app, join_app
import sys
sys.path.append(os.getcwd())

def pause_the_futures(pending_tasks, pending_task_limit):
    pending_tasks_num = len(pending_tasks)
    
    while pending_tasks_num >= pending_task_limit:
        print(f"There are {pending_tasks_num} pending tasks but a pending task limit of {pending_task_limit}")
        new_pending_tasks = []
        for t in pending_tasks:
            if t.running():
               new_pending_tasks.append(t)
        pending_tasks = new_pending_tasks
        pending_tasks_num = len(pending_tasks)
        time.sleep(10)            
    return pending_tasks, pending_tasks_num

def run_parsl(workflow_json="uniform_mini_sleep.json", pending_task_limit=10000):
    
    all_parsl_tasks = []
    pending_tasks = []
    pending_tasks_num = 0
    with open(workflow_json, "r") as f:
        workflow = json.load(f)
        tasks = workflow["tasks"]
        for task_set in tasks:
            config = set_config(provider="PBS")
            parsl.load(config)
            print('Loaded parsl config')
            #print(config)
            for i in range(tasks[task_set]["ntasks"]):
                args = tasks[task_set]["args"]
                t = parsl_sleeper(os.getcwd(), args)
                all_parsl_tasks.append(t)
                pending_tasks.append(t)
                pending_tasks_num += 1
                if pending_tasks_num > pending_task_limit:
                    pending_tasks, pending_tasks_num = pause_the_futures(pending_tasks, pending_task_limit)
                    print(f"Parsl running {pending_tasks_num} tasks out of {len(all_parsl_tasks)}")
    print(f"Parsl running {pending_tasks_num} tasks out of {len(all_parsl_tasks)}")

    # results = []
    # for t in all_parsl_tasks:
    #     ret = t.result()
    #     results.append(ret)
    #     print(ret)
    return all_parsl_tasks

# config = set_config()
# parsl.load(config)
# wf = run_parsl()
# print(wf.result())