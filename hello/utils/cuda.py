import torch
import torch.utils.benchmark as benchmark


def empty_cache():
    torch.cuda.empty_cache()


def zeros(size=1024, device="cuda:0", times=500):
    x = []
    for _ in range(times):
        x.append(torch.zeros(size, size, size, dtype=torch.int8, device=device))
        print(f"memory allocated: {torch.cuda.memory_allocated()/(1024**3):04} GB")


def matmul(size=1024, device="cuda:0", times=500):
    x = torch.randn(size, size, device=device)
    for _ in range(times):
        torch.matmul(x, x)


def timeit(size=1024, device="cuda:0", times=500):
    x = torch.randn(size, size, device=device)
    t = benchmark.Timer(
        stmt="torch.matmul(x, x)",
        setup="import torch",
        globals={"x": x},
    )
    return t.timeit(times)
