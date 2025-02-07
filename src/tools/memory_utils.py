import torch
import psutil

from typing import Dict


def get_memory_stats() -> Dict[str, float]:
    """
    Get current memory statistics for both CPU and GPU.

    Returns:
        Dict containing memory usage statistics in GB:
            - cpu_used: Current CPU memory used
            - cpu_available: Available CPU memory
            - gpu_used: Current GPU memory used (if available)
            - gpu_reserved: Total GPU memory reserved (if available)
    """
    memory_stats = {}

    # CPU Memory
    process = psutil.Process()
    cpu_memory = process.memory_info().rss / (1024 * 1024 * 1024)
    memory_stats["cpu_used"] = round(cpu_memory, 3)
    memory_stats["cpu_available"] = round(
        psutil.virtual_memory().available / (1024 * 1024 * 1024), 3
    )

    # GPU Memory
    if torch.cuda.is_available():
        # Convert to GB
        gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
        gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)
        gpu_cached = torch.cuda.memory_cached() / (1024 * 1024 * 1024)

        memory_stats["gpu_used"] = round(gpu_memory, 3)
        memory_stats["gpu_reserved"] = round(gpu_reserved, 3)
        memory_stats["gpu_cached"] = round(gpu_cached, 3)

    return memory_stats
