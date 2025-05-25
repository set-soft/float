import os
import psutil
import torch

def print_gpu_total_free_memory(device_id=0):
	if not torch.cuda.is_available():
		print("CUDA is not available.")
		return

	device = torch.device(f'cuda:{device_id}')
	if device_id >= torch.cuda.device_count():
		print(f"CUDA device {device_id} is not available. Max device ID: {torch.cuda.device_count()-1}")
		return

	free_bytes, total_bytes = torch.cuda.mem_get_info(device)

	total_mb = total_bytes / (1024 * 1024)
	free_mb = free_bytes / (1024 * 1024)
	used_mb = (total_bytes - free_bytes) / (1024 * 1024)
	return f" {used_mb:.2f}/{total_mb:.2f} MB"


def print_ram_usage():
	# Get the current process
	process = psutil.Process(os.getpid())

	# Get memory info (rss is a good measure of actual physical memory used)
	# rss: Resident Set Size - the portion of the process's memory held in RAM.
	# vms: Virtual Memory Size - total virtual address space used by the process.
	mem_info = process.memory_info()
	rss_bytes = mem_info.rss
	vms_bytes = mem_info.vms

	# Convert to a more readable format (e.g., MB)
	rss_mb = rss_bytes / (1024 * 1024)
	vms_mb = vms_bytes / (1024 * 1024)
	return f"{rss_mb:.2f}/{vms_mb:.2f} MB"