"""Sample GPU 0 through NVML into a CSV until killed. Started by common.GpuClockLog.

    python gpu_sampler.py out.csv 20        # every 20 ms

A separate process, so it never competes with the benchmark for the Python GIL. Uses
ctypes on libnvidia-ml (installed with the driver), no extra package needed.
"""
import ctypes
import sys
import time


def check(code, what):
    if code != 0:
        raise RuntimeError(f"NVML {what} failed with code {code}")


def main():
    path, interval = sys.argv[1], float(sys.argv[2]) / 1000
    nvml = ctypes.CDLL("nvml.dll" if sys.platform == "win32" else "libnvidia-ml.so.1")
    check(nvml.nvmlInit_v2(), "init")
    handle = ctypes.c_void_p()
    check(nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)), "handle")
    sm, mem, power, temp = (ctypes.c_uint() for _ in range(4))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("t,sm_mhz,mem_mhz,power_w,temp_c\n")
        while True:
            check(nvml.nvmlDeviceGetClockInfo(handle, 1, ctypes.byref(sm)), "SM clock")
            check(nvml.nvmlDeviceGetClockInfo(handle, 2, ctypes.byref(mem)), "memory clock")
            # Power and temperature are extra information: 0 if the driver refuses them.
            nvml.nvmlDeviceGetPowerUsage(handle, ctypes.byref(power))
            nvml.nvmlDeviceGetTemperature(handle, 0, ctypes.byref(temp))
            fh.write(f"{time.time():.4f},{sm.value},{mem.value},{power.value / 1000:.1f},{temp.value}\n")
            fh.flush()
            time.sleep(interval)


if __name__ == "__main__":
    main()
