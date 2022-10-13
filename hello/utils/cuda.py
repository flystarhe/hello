import timeit

import torch


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


def timeit_copy(size=1024, device="cuda:0", times=500):
    x = torch.zeros(size, size, size, dtype=torch.int8, device="cpu")
    t = timeit.Timer(
        stmt=f"x.to('{device}')",
        setup="import torch",
        globals={"x": x},
    )
    print(f"{t.timeit(times) / times * 1e3:>5.3f} ms")


def timeit_matmul(size=1024, device="cuda:0", times=500):
    x = torch.randn(size, size, device=device)
    t = timeit.Timer(
        stmt="torch.matmul(x, x)",
        setup="import torch",
        globals={"x": x},
    )
    print(f"{t.timeit(times) / times * 1e3:>5.3f} ms")
