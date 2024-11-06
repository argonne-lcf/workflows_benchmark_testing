import sys
import torch
try:
    import intel_extension_for_pytorch as ipex
except ModuleNotFoundError as e:
    pass
from torch.profiler import profile, record_function, ProfilerActivity

from sim_utils import gmres


if __name__ == "__main__":
    N = 512
    device = 'cpu'
    max_iter = 200
    restart = 50
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
        device = sys.argv[2]
    torch_device = torch.device(device)

    A = torch.randn(N, N, device=torch_device, dtype=torch.float64)
    b = torch.randn(N, device=torch_device, dtype=torch.float64)
    x, res, iters = gmres(
                          A, b,
                          tol=1e-7,
                          max_iter=max_iter,
                          restart=min(N,restart)
    )

    skip = 1
    wait = 0
    warmup = 1
    active = 1
    activities = [ProfilerActivity.CPU]
    if device=='cuda':
        activities.append(ProfilerActivity.CUDA)
    elif device=='xpu':
        activities.append(ProfilerActivity.XPU)
    #with profile(
    #            activities=activities,
    #            schedule=torch.profiler.schedule(
    #                skip_first=skip,
    #                wait=wait,
    #                warmup=warmup,
    #                active=active),
    #            record_shapes=True,
    #            profile_memory=True,
    #            with_stack=True,
    #            #on_trace_ready=trace_handler
    #    ) as prof:
    with profile(activities=activities, record_shapes=True, profile_memory=True, with_stack=True) as prof:

        A = torch.randn(N, N, device=torch_device, dtype=torch.float64)
        b = torch.randn(N, device=torch_device, dtype=torch.float64)
        x, res, iters = gmres(
                          A, b,
                          tol=1e-7,
                          max_iter=max_iter,
                          restart=min(N,restart)
        )

        #prof.step()

      
    print(prof.key_averages().table(sort_by=f"self_cpu_time_total", row_limit=10))
    print("\n\n\n")
    print(prof.key_averages().table(sort_by=f"self_{device}_time_total", row_limit=10))
    torch.save(prof.key_averages(), f'gmres_{device}_profile.pt')
    prof.export_chrome_trace(f"gmres_{device}_trace.json")



