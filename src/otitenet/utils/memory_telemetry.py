import torch

# Global tracker for peak GPU memory during training
_gpu_peak_tracker = {}


def _to_mb(num_bytes):
    return float(num_bytes) / (1024.0 * 1024.0)


def _device_index(device):
    if not torch.cuda.is_available():
        return 0
    if isinstance(device, str) and ":" in device:
        try:
            return int(device.split(":", 1)[1])
        except Exception:
            return torch.cuda.current_device()
    return torch.cuda.current_device()


def reset_gpu_peak(device="cuda:0"):
    """Reset peak memory tracking."""
    if not torch.cuda.is_available():
        return
    idx = _device_index(device)
    torch.cuda.synchronize(idx)
    torch.cuda.reset_peak_memory_stats(idx)
    torch.cuda.empty_cache()
    _gpu_peak_tracker[idx] = 0.0


def record_gpu_peak(device="cuda:0"):
    """Update the tracked peak GPU memory using true OS-visible usage (total - free)."""
    if not torch.cuda.is_available():
        return
    idx = _device_index(device)
    torch.cuda.synchronize(idx)
    free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
    used_mb = _to_mb(total_bytes - free_bytes)
    _gpu_peak_tracker[idx] = max(_gpu_peak_tracker.get(idx, 0.0), used_mb)


def get_gpu_peak_mb(device="cuda:0"):
    """Get the peak memory recorded so far (from record_gpu_peak)."""
    idx = _device_index(device)
    return _gpu_peak_tracker.get(idx, 0.0)


def gpu_memory_stats(device="cuda:0"):
    if not torch.cuda.is_available():
        return {}
    idx = _device_index(device)
    torch.cuda.synchronize(idx)
    free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
    # tracked_peak is total-free based; also compare with pytorch's internal max_memory_allocated
    tracked_peak = get_gpu_peak_mb(device)
    pytorch_peak = _to_mb(torch.cuda.max_memory_allocated(idx))
    # Prefer the tracked (total-free) peak; fall back to pytorch peak if tracker uninitialized
    actual_peak = max(tracked_peak, pytorch_peak)
    return {
        "total_mb": _to_mb(total_bytes),
        "free_mb": _to_mb(free_bytes),
        "used_mb": _to_mb(total_bytes - free_bytes),
        "allocated_mb": _to_mb(torch.cuda.memory_allocated(idx)),
        "reserved_mb": _to_mb(torch.cuda.memory_reserved(idx)),
        "actual_peak_gpu_mb": actual_peak,
    }


def estimate_theoretical_gpu_required_mb(model, batch_size, image_size, channels=3):
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    grad_and_opt_multiplier = 4.0
    model_footprint = param_bytes * grad_and_opt_multiplier
    activation_bytes = float(batch_size) * float(channels) * float(image_size) * float(image_size) * 4.0 * 16.0
    return _to_mb(model_footprint + activation_bytes)


def emit_gpu_telemetry(event, **fields):
    parts = [f"event={event}"]
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    print("OTITENET_GPU_TELEMETRY|" + "|".join(parts), flush=True)
