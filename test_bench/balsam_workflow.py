import json
import os
from balsam.api import Job, BatchJob, EventLog, Site

def run_balsam(workflow_json="uniform_mini_sleep.json", gpus_per_node=4):

    site_id = Site.objects.get("benchmark").id
    
    all_balsam_tasks = []
    with open(workflow_json, "r") as f:
        workflow = json.load(f)
        tasks = workflow["tasks"]
        for task_set in tasks:
            #print(f"{task_set=}",task_set['ntasks'])
            for i in range(tasks[task_set]["ntasks"]):
                params = {"app_path": os.getcwd(),
                          "args": tasks[task_set]['args']}
                
                num_nodes = tasks[task_set]['nodes_per_task']
                ranks_per_node = tasks[task_set]['ranks_per_task']//num_nodes
                threads_per_rank = tasks[task_set]['cpu_threads_per_task']//tasks[task_set]['ranks_per_task']
                threads_per_core = 2
                gpus_per_rank = tasks[task_set]['gpus_per_task']//tasks[task_set]['ranks_per_task']
                node_packing_count = max(1, gpus_per_node//tasks[task_set]['gpus_per_task'])
                j = Job(app_id="BenchmarkApp",
                        site_name="benchmark",
                        workdir=f"bench/{i}",
                        parameters=params,
                        num_nodes=num_nodes,
                        ranks_per_node=ranks_per_node,
                        threads_per_rank=threads_per_rank,
                        threads_per_core=threads_per_core,
                        gpus_per_rank=gpus_per_rank,
                        node_packing_count=node_packing_count)
                all_balsam_tasks.append(j)
    all_balsam_tasks = Job.objects.bulk_create(all_balsam_tasks)

    BatchJob.objects.create(
        site_id=site_id,
        num_nodes=1,
        wall_time_min=60,
        job_mode="serial",
        project="datascience",
        queue="debug"
    )

    results = []
    print("Waiting for tasks to complete")
    for t in all_balsam_tasks:
        results.append(t.result())
    return results

def query_jobs():
    ev = EventLog.objects.filter(job_id=[33994749])
    print(ev)



    
