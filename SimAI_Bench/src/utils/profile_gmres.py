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

    skip = 1
    wait = 0
    warmup = 1
    active = 1
    activities = [ProfilerActivity.CPU]
    if device=='cuda':
        activities.append(ProfilerActivity.CUDA)
    elif device=='xpu':
        activities.append(ProfilerActivity.XPU)
    with profile(
                activities=activities,
                schedule=torch.profiler.schedule(
                    skip_first=skip,
                    wait=wait,
                    warmup=warmup,
                    active=active),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                #on_trace_ready=trace_handler
        ) as prof:

        A = torch.randn(N, N, device=torch_device, dtype=torch.float64)
        b = torch.randn(N, device=torch_device, dtype=torch.float64)
        x, res, iters = gmres(
                          A, b,
                          tol=1e-7,
                          max_iter=max_iter,
                          restart=min(N,restart)
        )

        prof.step()

      
    #print(prof.key_averages().table(sort_by=f"self_{device}_time_total", row_limit=20))
    print(prof.key_averages().table())
    if device=='cuda':
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
    elif device=='xpu':
        print(prof.key_averages().table(sort_by="self_xpu_time_total", row_limit=20))
    else:
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    torch.save(prof.key_averages(), f'gmres_{device}_profile.tar')
    prof.export_chrome_trace(f"gmres_{device}_trace.json")



