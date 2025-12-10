

# Now moved to run_benchmark.sh

# Restricts numpy to 1 thread, as otherwise OS scheduler breaks:
#	numpy uses 12 threads for matrix-vector multiplication
#	MPI tries to use these threads aswell
#	OS Scheduler fights with itself and introduces stopping & starting
#	We must lock numpy and MKL to 1 thread so the scheduler is happy
#	Each CPU core will run exactly 1 matrix operation (N times)

#import os
#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"
#os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from mpi_timer import timer
from datetime import datetime
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
	now = datetime.now().strftime("%H:%M %d/%m%Y")
	with open("mpi_log.txt", "a") as f:
		f.write(f"\n== {size} ranks, {now} ==\n")

# waits for all ranks to catch up
comm.Barrier()


N = 100_000
size = 512

alpha = 3.0
beta = -1

A = np.random.rand(size, size).astype(np.float32)
B = np.random.rand(size).astype(np.float32)
C = np.random.rand(size).astype(np.float32)

A_T = A.T

@timer
def GEMV(alpha, beta, A, B, C, N):

	# Perform General Matrix-Vector Multiplication (GEMV)
	for _ in range(N):
		y = alpha * np.matmul(A, B) + beta * C

	# Outputs the final flattened matrix
	return y.flatten()

GEMV(alpha, beta, A, B, C, N)


# mpirun --bind-to core -n 4 python3 benchmark.py
