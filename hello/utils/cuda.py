import timeit

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


def timeit_copy(size=1024, device="cuda:0", times=500, mode="benchmark"):
    torch.cuda.set_device(device)

    x = torch.zeros(size, size, size, dtype=torch.int8, device="cpu")

    if mode.lower() == "benchmark":
        print("[INFO] Benchmarking with torch.utils.benchmark.Timer")
        m = benchmark
    else:
        print("[INFO] Benchmarking with timeit.Timer")
        m = timeit

    t = m.Timer(
        stmt="x.to('cuda')",
        setup="import torch",
        globals={"x": x},
    )

    if mode.lower() == "benchmark":
        print(t.timeit(times))
    else:
        print(f"{t.timeit(times) / times * 1e6:>5.3f} us")


def timeit_matmul(size=1024, device="cuda:0", times=500, mode="benchmark"):
    torch.cuda.set_device(device)

    x = torch.randn(size, size, dtype=torch.float32, device="cuda")

    if mode.lower() == "benchmark":
        print("[INFO] Benchmarking with torch.utils.benchmark.Timer")
        m = benchmark
    else:
        print("[INFO] Benchmarking with timeit.Timer")
        m = timeit

    t = m.Timer(
        stmt="torch.matmul(x, x)",
        setup="import torch",
        globals={"x": x},
    )

    if mode.lower() == "benchmark":
        print(t.timeit(times))
    else:
        print(f"{t.timeit(times) / times * 1e6:>5.3f} us")
