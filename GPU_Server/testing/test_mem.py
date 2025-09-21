#!/usr/bin/env python3
"""
GPU, CPU, and Memory Usage Monitor
Monitors system resources on a server with multiple NVIDIA GPUs
Checks usage every 5 seconds using PyTorch for GPU monitoring
"""

import torch
import psutil
import time
import datetime
from typing import Dict, List


def get_gpu_info() -> List[Dict]:
    """Get information about all available GPUs"""
    gpu_info = []
    
    if not torch.cuda.is_available():
        print("CUDA is not available. No GPUs detected.")
        return gpu_info
    
    num_gpus = torch.cuda.device_count()
    
    for i in range(num_gpus):
        try:
            # Get GPU properties
            gpu_props = torch.cuda.get_device_properties(i)
            
            # Get memory info
            torch.cuda.set_device(i)
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_reserved = torch.cuda.memory_reserved(i)
            memory_total = gpu_props.total_memory
            
            # Calculate memory usage percentage
            memory_used_percent = (memory_reserved / memory_total) * 100
            
            # Get GPU utilization (approximation based on memory usage)
            # Note: PyTorch doesn't provide direct GPU utilization
            # For more accurate GPU utilization, you might want to use nvidia-ml-py
            
            gpu_info.append({
                'id': i,
                'name': gpu_props.name,
                'memory_total_gb': memory_total / (1024**3),
                'memory_allocated_gb': memory_allocated / (1024**3),
                'memory_reserved_gb': memory_reserved / (1024**3),
                'memory_used_percent': memory_used_percent,
                'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
            })
            
        except Exception as e:
            print(f"Error getting info for GPU {i}: {e}")
    
    return gpu_info


def get_cpu_info() -> Dict:
    """Get CPU usage information"""
    return {
        'usage_percent': psutil.cpu_percent(interval=1),
        'cores_physical': psutil.cpu_count(logical=False),
        'cores_logical': psutil.cpu_count(logical=True),
        'frequency_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 0
    }


def get_memory_info() -> Dict:
    """Get system memory information"""
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'used_gb': memory.used / (1024**3),
        'used_percent': memory.percent,
        'swap_total_gb': swap.total / (1024**3),
        'swap_used_gb': swap.used / (1024**3),
        'swap_used_percent': swap.percent
    }


def print_system_stats():
    """Print formatted system statistics"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"System Monitor - {timestamp}")
    print(f"{'='*60}")
    
    # GPU Information
    print("\n GPU Information:")
    gpu_info = get_gpu_info()
    
    if gpu_info:
        for gpu in gpu_info:
            print(f"  GPU {gpu['id']}: {gpu['name']}")
            print(f"    Memory: {gpu['memory_used_percent']:.1f}% "
                  f"({gpu['memory_reserved_gb']:.2f}GB / {gpu['memory_total_gb']:.2f}GB)")
            print(f"    Allocated: {gpu['memory_allocated_gb']:.2f}GB")
            print(f"    Compute Capability: {gpu['compute_capability']}")
    else:
        print("  No GPUs detected or CUDA not available")
    
    # CPU Information
    print("\n CPU Information:")
    cpu_info = get_cpu_info()
    print(f"  Usage: {cpu_info['usage_percent']:.1f}%")
    print(f"  Cores: {cpu_info['cores_physical']} physical, {cpu_info['cores_logical']} logical")
    if cpu_info['frequency_mhz'] > 0:
        print(f"  Frequency: {cpu_info['frequency_mhz']:.0f} MHz")
    
    # Memory Information
    print("\n Memory Information:")
    memory_info = get_memory_info()
    print(f"  RAM: {memory_info['used_percent']:.1f}% "
          f"({memory_info['used_gb']:.2f}GB / {memory_info['total_gb']:.2f}GB)")
    print(f"  Available: {memory_info['available_gb']:.2f}GB")
    
    if memory_info['swap_total_gb'] > 0:
        print(f"  Swap: {memory_info['swap_used_percent']:.1f}% "
              f"({memory_info['swap_used_gb']:.2f}GB / {memory_info['swap_total_gb']:.2f}GB)")


def main():
    """Main monitoring loop"""
    print(" Starting System Resource Monitor")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            print_system_stats()
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\n Monitoring stopped by user")
    except Exception as e:
        print(f"\n Error occurred: {e}")


if __name__ == "__main__":
    main()
