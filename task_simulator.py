from operator import itemgetter
import random

# Routine to simulate packing tasks of varying sizes and durations on a set number of nodes
# Tasks execute in order of passed task list
def packing_simulator(tasks, max_nodes = 100, max_time=24.,use_t_est=True, verbose=False):
    
    available_nodes = max_nodes
    running_tasks = 0
    run_time = 0
    completed_tasks = 0
    uncompleteable_tasks = 0

    t_sim = 0
    dt = 1/60.

    while t_sim <= max_time and completed_tasks+uncompleteable_tasks < len(tasks):
        for task in tasks:
            if task["status"] == "ready" and t_sim < max_time:
                if available_nodes >= task["num_nodes"]:
                    if use_t_est:
                        if max_time - t_sim < task["t_est"]:
                            task["status"] = "uncompleteable"
                            uncompleteable_tasks += 1
                            continue
                    available_nodes -= task["num_nodes"]
                    task["status"] = "running"
                    task["start_time"] = t_sim
                    running_tasks += 1
                    #if verbose:
                    #    print(f"time {t_sim}: Starting task {task}")
            elif task["status"] == "running":
                if t_sim - task["start_time"] >= task["t_runtime"]:
                    task["status"] = "completed"
                    task["end_time"] = t_sim
                    running_tasks -= 1
                    available_nodes += task["num_nodes"]
                    run_time += (task["end_time"] - task["start_time"])*task["num_nodes"]
                    completed_tasks += 1
        if verbose:
            print(f"time {t_sim}: {running_tasks} running tasks, {completed_tasks} completed tasks, {available_nodes} idle nodes")
        t_sim += dt
    
    wall_time = t_sim - dt
    number_timed_out = 0
    for task in tasks:
        if task["status"] == "running":
            task["status"] = "timed_out"
            run_time += (wall_time-task["start_time"])*task["num_nodes"]
            number_timed_out += 1

    
    idle_time = wall_time*max_nodes - run_time

    print(f"Total node hours used: {run_time}")
    print(f"Total Walltime: {wall_time}")
    print(f"Fraction of idle time: {round(idle_time/(wall_time*max_nodes),2)}")
    print(f"tasks completed/timed_out/not_started: {completed_tasks}/{number_timed_out}/{len(tasks)-number_timed_out-completed_tasks}")

    return wall_time*max_nodes, run_time, idle_time/(wall_time*max_nodes), number_timed_out

def add_runtime_scatter(tasks, percent=0.1):
    for task in tasks:
        task["t_runtime"] = max(task["t_est"]*(1.+percent*random.uniform(-1.,1.)), 1/60.)
    return tasks

def make_tasks(n, t_est, ntasks, sort_option="default_balsam",runtime_scatter=True):
    tasks = []
    nbins = len(n)

    for i in range(nbins):
        for j in range(ntasks[i]):
            tasks.append({"num_nodes":n[i],
                        "t_est": t_est[i],
                        "t_runtime": t_est[i],
                        "status": "ready"})
    if runtime_scatter:
        tasks = add_runtime_scatter(tasks)

    if sort_option == "default_balsam":
        tasks = sorted(tasks, key=itemgetter("t_est"))
        sorted_tasks = sorted(tasks, key=itemgetter("t_est"))
        bin_t_est = tasks[0]["t_est"]
        bin_start_index = 0
        bin_tasks = []
     
        for i,task in enumerate(tasks):
        
            if task["t_est"] == bin_t_est and i < len(tasks) - 1:
                bin_end_index = i+1
                bin_tasks.append(task)
            else:
                bin_t_est = task["t_est"]
                bin_end_index =i
                if i == len(tasks)-1:
                    bin_end_index = i+1
                    bin_tasks.append(task)
                sorted_tasks[bin_start_index:bin_end_index] = sorted(bin_tasks,key=itemgetter("num_nodes"),reverse=True)
                bin_start_index = i
                bin_tasks = [task]
        
        print(len(tasks),len(sorted_tasks))
    elif sort_option == "long_large_first":
        tasks = sorted(tasks, key=itemgetter("t_est"),reverse=True)
        sorted_tasks = sorted(tasks, key=itemgetter("t_est"),reverse=True)
        bin_t_est = tasks[0]["t_est"]
        bin_start_index = 0
        bin_end_index = 1
        bin_tasks = []
        
        for i,task in enumerate(tasks):
            if task["t_est"] == bin_t_est and i < len(tasks) - 1:
                bin_end_index = i+1
                bin_tasks.append(task)
            else:
                
                bin_t_est = task["t_est"]
                bin_end_index = i
                if i == len(tasks)-1:
                    bin_end_index = i+1
                    bin_tasks.append(task)
                #print("Next bin",i,task["t_est"],bin_start_index,bin_end_index,len(bin_tasks))
                sorted_tasks[bin_start_index:bin_end_index] = sorted(bin_tasks,key=itemgetter("num_nodes"),reverse=True)
                bin_start_index = i
                bin_tasks = [task]

        print(len(tasks),len(sorted_tasks))
  
    return sorted_tasks

def make_linear_dist_tasks(n = [1,4,8,10],t_est = [0.5,8.,16.,22.],Ntot=100,flarge=0.1,sort_option="default_balsam"):

    nbins = len(n) # number of differently sized tasks

    sum_n = sum(n)
    nlarge = max(n)

    m = flarge*Ntot * (nlarge + (nlarge -flarge*sum_n)/(flarge*nbins - 1))**(-1)
    b = m*(nlarge - flarge*sum_n)/(flarge*nbins - 1)

    ntasks = [int(round(m*ni+b)) for ni in n]

    print(f"Task sizes: {n}")
    print(f"Task runtime estimates: {t_est}")
    print(f"Number of tasks: {ntasks}")

    return make_tasks(n, t_est, ntasks,sort_option=sort_option)
    
def make_dale_tasks(Ntot = 100, fsmall = 0.25, sort_option="default_balsam"):
    n = [2,4,4,4] # task node sizes
    t_est = [0.5,8.,16.,22.] # task run times

    nbins = len(n)

    nsmall = fsmall*Ntot

    ntasks = []
    for i in range(nbins):
        if n[i] == min(n):
            ntasks.append(int(round(nsmall)))
        if n[i] == max(n):
            ntasks.append(int(round((Ntot-nsmall)/(nbins-1))))

    print(ntasks,n,t_est,sum(ntasks))

    print(f"Task sizes: {n}")
    print(f"Task runtime estimates: {t_est}")
    print(f"Number of tasks: {ntasks}")

    return make_tasks(n, t_est, ntasks,sort_option=sort_option)
    
    
if __name__ == '__main__':

    max_nodes_polaris = 496
    max_nodes_aurora = 10000

    Ntot = 100

    for max_nodes in [200,max_nodes_polaris,max_nodes_aurora]:
        print(f"Linear Task Distribution Test: Total Nodes={max_nodes} Total Tasks={Ntot}")
        tasks = make_linear_dist_tasks(Ntot=Ntot,sort_option="long_large_first")
        out = packing_simulator(tasks, max_nodes = 200, max_time=24.)
        print("")

        print(f"Dale VASP distribution test: Total Nodes={max_nodes} Total Tasks={Ntot}")
        tasks = make_dale_tasks(Ntot=Ntot, sort_option="long_large_first")
        out = packing_simulator(tasks, max_nodes = 200, max_time=24.)
        print("")

