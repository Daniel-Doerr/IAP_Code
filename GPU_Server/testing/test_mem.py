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
import subprocess
import json
import os
from typing import Dict, List


def get_gpu_info_nvidia_smi() -> List[Dict]:
    """Get GPU information using nvidia-smi command for accurate VRAM usage"""
    gpu_info = []
    
    try:
        # Run nvidia-smi with JSON output for detailed GPU information
        cmd = [
            'nvidia-smi', 
            '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,compute_cap',
            '--format=csv,noheader,nounits'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [part.strip() for part in line.split(',')]
                if len(parts) >= 10:
                    gpu_id = int(parts[0])
                    name = parts[1]
                    memory_total = float(parts[2])  # MB
                    memory_used = float(parts[3])   # MB  
                    memory_free = float(parts[4])   # MB
                    gpu_util = float(parts[5]) if parts[5] != '[Not Supported]' else 0
                    mem_util = float(parts[6]) if parts[6] != '[Not Supported]' else 0
                    temperature = float(parts[7]) if parts[7] != '[Not Supported]' else 0
                    power_draw = float(parts[8]) if parts[8] != '[Not Supported]' else 0
                    compute_cap = parts[9]
                    
                    memory_used_percent = (memory_used / memory_total) * 100
                    
                    gpu_info.append({
                        'id': gpu_id,
                        'name': name,
                        'memory_total_gb': memory_total / 1024,
                        'memory_used_gb': memory_used / 1024,
                        'memory_free_gb': memory_free / 1024,
                        'memory_used_percent': memory_used_percent,
                        'gpu_utilization_percent': gpu_util,
                        'memory_utilization_percent': mem_util,
                        'temperature_c': temperature,
                        'power_watts': power_draw,
                        'compute_capability': compute_cap,
                        'source': 'nvidia-smi'
                    })
        
        # Get process information
        try:
            proc_cmd = ['nvidia-smi', 'pmon', '-c', '1', '-s', 'um']
            proc_result = subprocess.run(proc_cmd, capture_output=True, text=True, check=True)
            
            # Parse process information and add to GPU info
            process_info = {}
            for line in proc_result.stdout.strip().split('\n')[2:]:  # Skip header
                if line.strip() and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 4:
                        gpu_id = int(parts[0])
                        if gpu_id not in process_info:
                            process_info[gpu_id] = {'count': 0, 'memory_mb': 0}
                        process_info[gpu_id]['count'] += 1
                        try:
                            mem_mb = int(parts[3])
                            process_info[gpu_id]['memory_mb'] += mem_mb
                        except:
                            pass
            
            # Add process info to GPU data
            for gpu in gpu_info:
                gpu_id = gpu['id']
                if gpu_id in process_info:
                    gpu['process_count'] = process_info[gpu_id]['count']
                    gpu['process_memory_gb'] = process_info[gpu_id]['memory_mb'] / 1024
                else:
                    gpu['process_count'] = 0
                    gpu['process_memory_gb'] = 0
        except:
            # If process monitoring fails, set defaults
            for gpu in gpu_info:
                gpu['process_count'] = 0
                gpu['process_memory_gb'] = 0
                
    except subprocess.CalledProcessError:
        print("nvidia-smi not available, falling back to PyTorch monitoring")
        return get_gpu_info_pytorch()
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")
        return get_gpu_info_pytorch()
    
    return gpu_info


def get_gpu_info_pytorch() -> List[Dict]:
    """Fallback GPU monitoring using PyTorch (less accurate for VRAM)"""
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
            
            gpu_info.append({
                'id': i,
                'name': gpu_props.name,
                'memory_total_gb': memory_total / (1024**3),
                'memory_used_gb': memory_reserved / (1024**3),
                'memory_free_gb': (memory_total - memory_reserved) / (1024**3),
                'memory_used_percent': memory_used_percent,
                'gpu_utilization_percent': 0,
                'memory_utilization_percent': 0,
                'temperature_c': 0,
                'power_watts': 0,
                'process_count': 0,
                'process_memory_gb': memory_allocated / (1024**3),
                'compute_capability': f"{gpu_props.major}.{gpu_props.minor}",
                'source': 'pytorch'
            })
            
        except Exception as e:
            print(f"Error getting info for GPU {i}: {e}")
    
    return gpu_info


def get_gpu_info() -> List[Dict]:
    """Get information about all available GPUs using best available method"""
    return get_gpu_info_nvidia_smi()


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


def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def collect_all_system_data():
    """Collect all system data first, then return formatted output"""
    # Collect all data first
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu_info = get_gpu_info()
    cpu_info = get_cpu_info()
    memory_info = get_memory_info()
    
    # Build the complete output string
    output_lines = []
    
    # Header
    output_lines.append("="*70)
    output_lines.append(f"System Monitor - {timestamp}")
    output_lines.append("="*70)
    
    # GPU Information
    output_lines.append("")
    output_lines.append("GPU Information:")
    
    if gpu_info:
        for gpu in gpu_info:
            output_lines.append(f"  GPU {gpu['id']}: {gpu['name']}")
            output_lines.append(f"    VRAM Usage: {gpu['memory_used_percent']:.1f}% "
                              f"({gpu['memory_used_gb']:.2f}GB / {gpu['memory_total_gb']:.2f}GB)")
            output_lines.append(f"    VRAM Free: {gpu['memory_free_gb']:.2f}GB")
            
            if gpu.get('gpu_utilization_percent', 0) > 0:
                output_lines.append(f"    GPU Utilization: {gpu['gpu_utilization_percent']}%")
                output_lines.append(f"    Memory Utilization: {gpu['memory_utilization_percent']}%")
                
            if gpu.get('temperature_c', 0) > 0:
                output_lines.append(f"    Temperature: {gpu['temperature_c']}°C")
                
            if gpu.get('power_watts', 0) > 0:
                output_lines.append(f"    Power Draw: {gpu['power_watts']:.1f}W")
                
            if gpu.get('process_count', 0) > 0:
                output_lines.append(f"    Active Processes: {gpu['process_count']} "
                                  f"(Memory: {gpu['process_memory_gb']:.2f}GB)")
                                  
            if gpu.get('source') == 'pytorch':
                output_lines.append(f"    Note: PyTorch memory only (nvidia-smi not available)")
            elif gpu.get('compute_capability'):
                output_lines.append(f"    Compute Capability: {gpu['compute_capability']}")
                
            output_lines.append(f"    Data Source: {gpu.get('source', 'unknown')}")
            output_lines.append("")  # Empty line between GPUs
    else:
        output_lines.append("  No GPUs detected or CUDA not available")
    
    # CPU Information
    output_lines.append("CPU Information:")
    output_lines.append(f"  Usage: {cpu_info['usage_percent']:.1f}%")
    output_lines.append(f"  Cores: {cpu_info['cores_physical']} physical, {cpu_info['cores_logical']} logical")
    if cpu_info['frequency_mhz'] > 0:
        output_lines.append(f"  Frequency: {cpu_info['frequency_mhz']:.0f} MHz")
    
    # Memory Information
    output_lines.append("")
    output_lines.append("System Memory Information:")
    output_lines.append(f"  RAM Usage: {memory_info['used_percent']:.1f}% "
                       f"({memory_info['used_gb']:.2f}GB / {memory_info['total_gb']:.2f}GB)")
    output_lines.append(f"  RAM Available: {memory_info['available_gb']:.2f}GB")
    
    if memory_info['swap_total_gb'] > 0:
        output_lines.append(f"  Swap Usage: {memory_info['swap_used_percent']:.1f}% "
                           f"({memory_info['swap_used_gb']:.2f}GB / {memory_info['swap_total_gb']:.2f}GB)")
    
    # Footer
    output_lines.append("")
    output_lines.append("="*70)
    output_lines.append("Press Ctrl+C to stop monitoring | Updates every 5 seconds")
    output_lines.append("="*70)
    
    return "\n".join(output_lines)


def print_system_stats():
    """Print formatted system statistics at fixed position"""
    # Collect all data first, then display at once
    complete_output = collect_all_system_data()
    
    # Clear screen and display all at once
    clear_screen()
    print(complete_output)


def main():
    """Main monitoring loop"""
    # Initial clear and startup message
    clear_screen()
    print("Starting Advanced System Resource Monitor")
    print("Using nvidia-smi for accurate GPU VRAM monitoring")
    print("Initializing...")
    time.sleep(2)
    
    try:
        while True:
            print_system_stats()
            time.sleep(5)
            
    except KeyboardInterrupt:
        clear_screen()
        print("System Resource Monitor stopped by user")
    except Exception as e:
        clear_screen()
        print(f"Error occurred: {e}")
        print("Monitor stopped due to error")


if __name__ == "__main__":
    main()
