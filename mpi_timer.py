import time
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()

def timer(func):
	def wrapper(*args, **kwargs):
		start = time.perf_counter()
		result = str(func(*args, **kwargs))
		end = time.perf_counter()
		elapsed = end - start
		s = f"RANK {rank} | {func.__name__} took {elapsed:.6f}s | {result[:30]}" + ("..." if len(result) >= 30 else " ")
		with open("mpi_log.txt", "a") as f:
			f.write(s + "\n")
		print(s)
		return result
	return wrapper
