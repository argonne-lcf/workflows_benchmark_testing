import os
import subprocess
import time

def launch_task(task,assigned_nodes):
    print(f"Launching task on nodes {assigned_nodes}")

    num_nodes = len(assigned_nodes)
    ppn = 12
    nranks = num_nodes*ppn

    hosts_str = ""
    for node in assigned_nodes:
        if node != assigned_nodes[0]:
            hosts_str += ","
        hosts_str += node
    cmd = f"mpiexec -n {nranks} --ppn {ppn} --hosts {hosts_str} sleep 10"
    p = subprocess.Popen(cmd, shell=True)

    return p
    
ntasks = 10

# Make simple task list
tasks = {}
for i in range(ntasks):
    tasks[str(i)] = { 'type':'lammps',
                      'num_nodes': 2,
                      'status': 'ready'}

# Get available nodes
free_node_list = []
node_file = os.getenv("PBS_NODEFILE")
with open(node_file,"r") as f:
    free_node_list = f.readlines()
    free_node_list = [node.split("\n")[0] for node in free_node_list]
total_job_nodes = len(free_node_list)

ready_tasks_id = list(tasks.keys())
running_tasks_id = []
num_unfinished_tasks = len(ready_tasks_id)
launched_procs = {}
while num_unfinished_tasks > 0:

    # Launch ready tasks
    for tid in ready_tasks_id:

        if tasks[tid]["num_nodes"] <= len(free_node_list):
            assigned_nodes = free_node_list[0:tasks[tid]["num_nodes"]]
            print(f"Launching task {tid}")
            p = launch_task(tasks[tid],assigned_nodes=assigned_nodes)
            tasks[tid]['status'] = "running"
            ready_tasks_id.remove(tid)
            running_tasks_id.append(tid)
            launched_procs[tid] = {"process":p,
                                   "assigned_nodes": assigned_nodes}
            for node in assigned_nodes:
                free_node_list.remove(node)

    # Poll running tasks
    for tid in running_tasks_id:
        popen_proc = launched_procs[tid]["process"]
        if popen_proc.poll() is not None:
            print(f"Task {tid} returned")
            if popen_proc.returncode == 0:
                tasks[tid]["status"] = "finished"
            else:
                tasks[tid]["status"] = "failed"
            running_tasks_id.remove(tid)
            assigned_nodes = launched_procs[tid]["assigned_nodes"]
            for node in assigned_nodes:
                free_node_list.append(node)
            num_unfinished_tasks -= 1
    print(f"Nodes occupied {total_job_nodes - len(free_node_list)}/{total_job_nodes}")
    print(f"Tasks ready: {len(ready_tasks_id)}")
    print(f"Tasks running: {len(running_tasks_id)}")
    time.sleep(1)
print(tasks)
    
    
            
