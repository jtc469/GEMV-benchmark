# GEMV-benchmark


Small MPI benchmark for a matrix vector multiply (GEMV) kernel, written in Python with `NumPy` and `mpi4py`.

I used this project to learn how MPI behaves on a single laptop using a Rocky Linux WSL environment, and to explore practical issues like oversubscription, BLAS threading, heterogeneous cores, and logging MPI runs.



---



## Background



I wanted to create a realistic HPC workload that I could run on my Rocky Linux WSL:



- A basic linear algebra kernel that maps onto BLAS routines.
- Multiple MPI ranks running on a single node.
- Per rank timings and logs for my report.
- All of this inside a Rocky Linux image running under WSL2 on a Windows laptop.



The point is to see how performance changes with:



- Number of MPI ranks vs physical cores.
- Matrix size and number of repetitions.
- BLAS threading settings.
- The topology of the CPU that sits underneath WSL.



## What the benchmark does



There are three files:
- `benchmark.py`
- `mpi_timer.py`
- `run_benchmark.sh`



### GEMV kernel



Each MPI rank runs the same function:



- Allocate a random dense matrix `A` of shape `(size, size)` and vectors `B` and `C` of length `size` in `float32`.

- Form `A_T = A.T` once.

- Repeat this `N` times:



$ y = \alpha A^{T} B + \beta C $



- Return the final `y` flattened.



Defaults in the code:



- `size = 512`

- `N = 100_000`

- `alpha = 3.0`

- `beta = -1.0`



There is also a 10 % chance that a rank will skip the computation and immediately return the string `"SKIPPED"`. This gives a simple artificial source of load imbalance, and shows how the total time is dependent on the slowest rank!



### Timing and logging



`mpi_timer.py` defines a decorator `@timer` that:



- Records `time.perf_counter()` before and after the function.

- Prints a line with:



&nbsp; `RANK <id> | <func> took <seconds>s | <first part of result>`



- Appends the same line to `mpi_log.txt`.



`benchmark.py` does some extra setup:



- Uses `mpi4py` to get the communicator size and rank.

- Rank 0 prints a header into `mpi_log.txt` for each run, for example



&nbsp; `== 4 ranks, 14:14 10/12/2025 ==`



- All ranks call `comm.Barrier()` so the header appears before any lines from the ranks.

- Applies `@timer` to the GEMV function and calls it once in `main`.



`run_benchmark.sh` is a tiny wrapper that runs:



`mpirun --bind-to core -n <num_ranks> python3 benchmark.py`



---



## Problems I hit and what I learnt



### 1. Oversubscription and “wrong” scaling



My first expectation was that if each rank does the same amount of work, the runtime per rank would be roughly constant when I increase the number of MPI ranks on a single machine.



That did not happen. As I increased `-n`, each rank became slower. Reasons:



- All ranks share the same memory bandwidth and last level cache.

- All ranks run on the same physical socket so they compete for execution units.

- When I went past the number of physical cores, the OS started time slicing more aggressively.



Lesson: on a fixed single node you should expect per rank runtime to increase past a certain number of ranks, especially for memory heavy kernels. For clean scaling experiments you either need more nodes, or you need to divide the global problem across ranks instead of duplicating it.



### 2. BLAS threading inside MPI ranks



Initially NumPy used its default BLAS threading configuration. That meant:



- Each MPI rank could spawn multiple BLAS threads.

- MPI and BLAS threads fought for cores.

- The OS scheduler swapped things around and runtimes were noisy.



I fixed this by setting the following environment variables in the shell before `mpirun`:



- `OMP_NUM_THREADS=1`

- `MKL_NUM_THREADS=1`

- `OPENBLAS_NUM_THREADS=1`



### 3. Heterogeneous cores and rank placement



Through this experiment I discovered my laptop CPU has performance cores and efficiency cores. With `mpirun --bind-to core -n 8` I noticed:



- Four ranks finished around one time (~3.8s).

- The other four ranks finished noticeably (~5.6 seconds).



All ranks ran the same code. The difference came from which logical CPUs they were bound to. Some ranks landed on performance cores, others on efficiency cores.



Lesson: even on a single node, hardware is not always homogeneous as core type and binding matter. MPI placement can give very different timings for identical work, which is important when interpreting per rank results. On a dedicated supercomputer or server, it's important to ensure that all cores are identical.



### 4. Thermal throttling



With very large matrices and many ranks I started to see longer and more variable runtimes. The CPU  power and temperature rose, then the CPU reduced its frequency.



Reducing the matrix size for example from `4096` down to `512` and increasing `N` to keep the total work large gave long runtimes but a lower, more stable power draw.



Lesson: for home experiments it is often better to use lower per-iteration work with greater iterations rather than one huge kernel that utilises all cores at once.



---



## How to run it



From inside the repository the basic steps are:



- Install `numpy` and `mpi4py` into your Python environment.

- Run `sh run_benchmark.sh 4` and compare the results with `sh run_benchmark.sh 8`



Each run prints timings to the terminal and appends them to `mpi_log.txt` together with a header showing the number of ranks and a timestamp.



You can also tweak the workload by changing `N` and `size` in `benchmark.py`.



---



## Possible extensions



Some ideas for future work:



- Plot per rank runtimes vs rank id for different `-n` values to visualise core heterogeneity.

- Repeat the benchmark test in C++ to compare with Python speeds (i suspect minimal improvement as NumPy is likely highly optimised in C)





