import os
import sys
import re
import json
import time
import hashlib
import platform
import threading
import subprocess
import numpy as np
from numba import njit, uint64

try:
    import resource
except ImportError:
    resource = None

# Replace with your exact Git handle before executing
GIT_USERNAME = "marzhansherekhan1823"
STUDENT_ID = "230109015"

NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000
STRIDES = (1, 4, 8, 16)
REPS = 3                      # repetitions per layout (median is reported)
GIL_EXPERIMENT_ITERS = 1_000_000
BUFFER_ALIGN = 128            # >= 64 B; 128 B so L2 adjacent-line pairs are deterministic


def detect_cache_line_size() -> int:
    """Query L1 data cache coherency line size in bytes from OS sysfs/sysctl."""
    linux_path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(linux_path):
        with open(linux_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    if sys.platform == "darwin":
        out = subprocess.check_output(["sysctl", "-n", "hw.cachelinesize"], encoding="utf-8")
        return int(out.strip())
    return 64


def get_cpu_model() -> str:
    """CPU model string (Linux /proc/cpuinfo, macOS sysctl, else platform)."""
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], encoding="utf-8").strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _parse_cpu_list(text: str) -> set:
    cpus = set()
    for part in text.strip().split(","):
        if "-" in part:
            a, b = part.split("-")
            cpus.update(range(int(a), int(b) + 1))
        elif part:
            cpus.add(int(part))
    return cpus


def pick_cores(n: int):
    """Pick n logical CPUs on distinct physical cores (avoid SMT siblings sharing an L1).
    Returns (list_of_cpus_or_None, all_on_distinct_physical_cores)."""
    if not hasattr(os, "sched_getaffinity"):
        return None, False
    avail = sorted(os.sched_getaffinity(0))
    chosen, used = [], set()
    for c in avail:
        sib = {c}
        p = f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    sib = _parse_cpu_list(f.read()) | {c}
            except Exception:
                pass
        if sib & used:
            continue
        chosen.append(c)
        used |= sib
        if len(chosen) == n:
            return chosen, True
    return avail[:n], False


# ==============================================================================
# TASK 1: Hardware Topology & Student Identity Context
# ==============================================================================
def init_system_context(git_user: str) -> dict:
    assert git_user != "REPLACE_WITH_YOUR_GIT_USERNAME" and len(git_user.strip()) > 0, (
        "Set GIT_USERNAME to your real Git handle before running."
    )
    logical_cores = os.cpu_count() or 1
    if logical_cores < NUM_THREADS:
        raise RuntimeError(f"Requires at least {NUM_THREADS} logical cores; detected {logical_cores}.")

    line_size = detect_cache_line_size()
    cpu_model = get_cpu_model()
    cpu_sig = f"{platform.machine()}::{platform.processor()}::{logical_cores}::{cpu_model}"

    digest = hashlib.sha256(f"{git_user.strip()}::{cpu_sig}".encode("utf8")).hexdigest()
    instance_seed = int(digest[:16], 16)

    cores, distinct = pick_cores(NUM_THREADS)
    return {
        "git_username": git_user.strip(),
        "student_id": STUDENT_ID,
        "logical_cores": logical_cores,
        "cache_line_bytes": line_size,
        "cpu_model": cpu_model,
        "cpu_signature": cpu_sig,
        "instance_seed": instance_seed,
        "pinned_cores": cores,
        "cores_on_distinct_physical": distinct,
        "affinity_supported": cores is not None and hasattr(os, "sched_setaffinity"),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


# ==============================================================================
# TASK 2: GIL-free native counter kernel (real load+store each iteration)
# ==============================================================================
@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, seed):
    """In-place RMW on buffer[idx]. The next iteration's address depends on the value
    just loaded (hi = v >> 63, always 0 at run time but unprovable to the compiler),
    so LLVM cannot promote buffer[slot] into a register: every iteration issues a
    real i64 load and i64 store."""
    inc = np.uint64(1) + (np.uint64(seed) & np.uint64(3))
    top = np.uint64(63)
    idx = slot
    for i in range(iters):
        v = buffer[idx]
        hi = v >> top
        buffer[idx] = v + inc + hi
        idx = slot + np.int64(hi)


# ==============================================================================
# TASK 4: Thread-local register privatization baseline
# ==============================================================================
@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, seed):
    """Same arithmetic as the strided kernel, but accumulated in a register;
    shared memory is written exactly once at the end."""
    inc = np.uint64(1) + (np.uint64(seed) & np.uint64(3))
    top = np.uint64(63)
    acc = np.uint64(0)
    for i in range(iters):
        acc = acc + inc + (acc >> top)
    buffer[slot] = acc


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def aligned_u64_array(n_elems: int, align: int = BUFFER_ALIGN) -> np.ndarray:
    raw = np.zeros(n_elems * 8 + align, dtype=np.uint8)
    off = (-raw.ctypes.data) % align
    arr = raw[off:off + n_elems * 8].view(np.uint64)
    assert arr.ctypes.data % align == 0 and arr.dtype == np.uint64
    return arr


def thread_seed(instance_seed: int, tid: int) -> int:
    return int((instance_seed ^ ((tid + 1) * 0x9E3779B1)) & 0xFFFFFFFF)


def expected_final(instance_seed: int, iters: int) -> list:
    out = []
    for t in range(NUM_THREADS):
        inc = 1 + (thread_seed(instance_seed, t) & 3)
        out.append(iters * inc)
    return out


def _thread_main(kernel, buf, slot, iters, seed, core, barrier, times, tid):
    if core is not None and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {core})   # pid 0 = calling thread on Linux
        except OSError:
            pass
    barrier.wait()
    t0 = time.perf_counter_ns()
    kernel(buf, slot, iters, seed)
    times[tid] = time.perf_counter_ns() - t0


def run_threads(ctx, kernel, buf, stride, iters):
    """Run NUM_THREADS native threads on buf with slot = tid*stride. Returns dict."""
    barrier = threading.Barrier(NUM_THREADS + 1)
    times = [0] * NUM_THREADS
    cores = ctx["pinned_cores"] if ctx["affinity_supported"] else [None] * NUM_THREADS
    threads = []
    for t in range(NUM_THREADS):
        th = threading.Thread(
            target=_thread_main,
            args=(kernel, buf, t * stride, iters, thread_seed(ctx["instance_seed"], t),
                  cores[t], barrier, times, t),
        )
        threads.append(th)
        th.start()
    ru0 = resource.getrusage(resource.RUSAGE_SELF) if resource else None
    barrier.wait()
    t0 = time.perf_counter_ns()
    for th in threads:
        th.join()
    elapsed = time.perf_counter_ns() - t0
    ru1 = resource.getrusage(resource.RUSAGE_SELF) if resource else None
    return {
        "elapsed_ns": elapsed,
        "thread_ns": list(times),
        "vol_ctx_switches": (ru1.ru_nvcsw - ru0.ru_nvcsw) if ru0 else None,
        "invol_ctx_switches": (ru1.ru_nivcsw - ru0.ru_nivcsw) if ru0 else None,
    }


def warm_up(ctx):
    tiny = aligned_u64_array(64)
    for k in (worker_strided_accumulate, worker_private_accumulate):
        k(tiny, 0, 1000, 1)


# ==============================================================================
# TASK 3: Multi-threaded stride sweep (+ private baseline run)
# ==============================================================================
def run_stride_benchmark(ctx, iters=None, reps=None):
    iters = iters or ITERATIONS_PER_THREAD
    reps = reps or REPS
    warm_up(ctx)

    runs = {s: [] for s in STRIDES}
    finals = {}
    for _ in range(reps):
        for s in STRIDES:
            buf = aligned_u64_array(NUM_THREADS * s + 32)
            res = run_threads(ctx, worker_strided_accumulate, buf, s, iters)
            res["final"] = [int(buf[t * s]) for t in range(NUM_THREADS)]
            runs[s].append(res)
            finals[s] = res["final"]

    # Task 4 baseline: register-private accumulation (shared slots, stride 1 layout)
    priv_runs = []
    priv_final = None
    for _ in range(reps):
        buf = aligned_u64_array(NUM_THREADS + 32)
        res = run_threads(ctx, worker_private_accumulate, buf, 1, iters)
        res["final"] = [int(buf[t]) for t in range(NUM_THREADS)]
        priv_runs.append(res)
        priv_final = res["final"]

    def summarize(rl):
        el = sorted(r["elapsed_ns"] for r in rl)
        med = el[len(el) // 2]
        return {
            "elapsed_ns_median": med,
            "elapsed_ns_min": el[0],
            "elapsed_ns_runs": [r["elapsed_ns"] for r in rl],
            "ns_per_update_per_thread": med / iters,
            "aggregate_ns_per_update": med / (iters * NUM_THREADS),
            "vol_ctx_switches": [r["vol_ctx_switches"] for r in rl],
            "invol_ctx_switches": [r["invol_ctx_switches"] for r in rl],
            "per_thread_ns_last_run": rl[-1]["thread_ns"],
        }

    results = {f"stride_{s}": dict(summarize(runs[s]), stride=s, stride_bytes=8 * s) for s in STRIDES}
    results["private_baseline"] = summarize(priv_runs)
    return results, finals, priv_final


def gil_experiment(ctx):
    """Plain Python bytecode threads (no nogil) on a stride-1 numpy buffer, for Q5."""
    def py_worker(buf, slot, n):
        for _ in range(n):
            buf[slot] += 1

    def run(nthreads):
        buf = aligned_u64_array(32)
        ths = [threading.Thread(target=py_worker, args=(buf, t, GIL_EXPERIMENT_ITERS))
               for t in range(nthreads)]
        t0 = time.perf_counter_ns()
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        return time.perf_counter_ns() - t0

    t1 = run(1)
    t4 = run(NUM_THREADS)
    return {
        "iters_per_thread": GIL_EXPERIMENT_ITERS,
        "one_thread_ns": t1,
        "four_threads_ns": t4,
        "slowdown_4_vs_1_thread": t4 / t1,
    }


# ==============================================================================
# TASK 5: Exact parity & LLVM IR verification
# ==============================================================================
def audit_llvm_and_parity(ctx, finals, priv_final, iters):
    expected = expected_final(ctx["instance_seed"], iters)
    parity = {f"stride_{s}": finals[s] == expected for s in STRIDES}
    parity["private_baseline"] = priv_final == expected
    layouts_equal = all(finals[s] == finals[STRIDES[0]] for s in STRIDES) and priv_final == finals[STRIDES[0]]
    parity_ok = all(parity.values()) and layouts_equal

    ir_map = worker_strided_accumulate.inspect_llvm()
    ir = "\n".join(ir_map.values()) if isinstance(ir_map, dict) else str(ir_map)
    sigs = [str(s) for s in worker_strided_accumulate.signatures]
    # Isolate the native nopython kernel body: the CPython wrapper (argument unboxing,
    # PyNumber_Long, GIL release) is a separate function and is not the compute loop.
    kernel_ir = ir
    for m in re.finditer(r'^define [^\n]*@"?(_ZN\d+\w*?worker_strided_accumulate\w*)"?\(.*?^}', ir, re.M | re.S):
        if not m.group(1).startswith(("_ZN7cpython", "cfunc")):
            kernel_ir = m.group(0)
            break
    llvm = {
        "store_i64_count": len(re.findall(r"\bstore i64\b", kernel_ir)),
        "load_i64_count": len(re.findall(r"\bload i64\b", kernel_ir)),
        "object_mode_calls": len(re.findall(r"PyObject_(Call|GetItem|SetItem)|PyNumber_", kernel_ir)),
        "signatures": sigs,
        "pyobject_in_signature": any("pyobject" in s.lower() for s in sigs),
    }
    llvm_ok = (llvm["store_i64_count"] > 0 and llvm["load_i64_count"] > 0
               and llvm["object_mode_calls"] == 0 and not llvm["pyobject_in_signature"])
    return {
        "expected_final_values": expected,
        "bit_exact_per_layout": parity,
        "layouts_identical": layouts_equal,
        "parity_ok": parity_ok,
        "llvm": llvm,
        "llvm_ok": llvm_ok,
    }, parity_ok and llvm_ok


def main():
    ctx = init_system_context(GIT_USERNAME)
    print("=== System context ===")
    for k in ("git_username", "student_id", "logical_cores", "cache_line_bytes", "cpu_model",
              "instance_seed", "pinned_cores", "cores_on_distinct_physical", "affinity_supported"):
        print(f"{k}: {ctx[k]}")
    if not ctx["cores_on_distinct_physical"]:
        print("WARNING: could not find 4 distinct physical cores (SMT siblings/VM vCPUs may share L1); "
              "false-sharing effects can be weaker.")

    print("\n=== Stride sweep (this takes a while) ===")
    results, finals, priv_final = run_stride_benchmark(ctx)
    base8 = results["stride_8"]["elapsed_ns_median"]
    for key in ("stride_1", "stride_4", "stride_8", "stride_16", "private_baseline"):
        r = results[key]
        print(f"{key:17s} median = {r['elapsed_ns_median']:>15,d} ns | "
              f"{r['ns_per_update_per_thread']:.3f} ns/update/thread | "
              f"vs stride_8: {r['elapsed_ns_median'] / base8:.3f}x | "
              f"ctx sw (vol/invol) = {r['vol_ctx_switches']}/{r['invol_ctx_switches']}")
    slowdown_1_vs_8 = results["stride_1"]["elapsed_ns_median"] / base8
    print(f"\nSlowdown stride_1 / stride_8 = {slowdown_1_vs_8:.3f}x")

    print("\n=== GIL experiment (no nogil) ===")
    gil = gil_experiment(ctx)
    print(gil)

    print("\n=== Parity & LLVM audit ===")
    audit, ok = audit_llvm_and_parity(ctx, finals, priv_final, ITERATIONS_PER_THREAD)
    print(json.dumps(audit, indent=2))

    manifest = {
        "context": ctx,
        "config": {"threads": NUM_THREADS, "iterations_per_thread": ITERATIONS_PER_THREAD,
                   "strides": list(STRIDES), "reps": REPS, "buffer_alignment_bytes": BUFFER_ALIGN},
        "results": results,
        "slowdown_stride1_vs_stride8": slowdown_1_vs_8,
        "gil_experiment": gil,
        "audit": audit,
        "verification_passed": bool(ok),
    }
    with open("cache_lab_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nverification_passed = {ok}")
    print("Wrote cache_lab_manifest.json")


if __name__ == "__main__":
    main()
