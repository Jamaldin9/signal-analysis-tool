import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt   # <-- missing import fixed

# <---- Load Data ---->
# Load data from a different path 
data = pd.read_csv("data/sample_signal.csv")

# create the columns necessary for plotting: time and signal
t = data["time"].to_numpy()
s = data["signal"].to_numpy()

# <----- Use a simple moving average filter to smooth the signal----->
# This smooths fast noise. It's a common first step in sensor processing.
# Our sample data steps by 0.01 seconds -> 100 samples per second (100 Hz)
fs = 100  # Sampling frequency in Hz

cutoff_hz = 5  # keep slower changes, remove fast wiggles
order = 4

# Design a Butterworth low-pass filter
nyquist = 0.5 * fs
normal_cutoff = cutoff_hz / nyquist
b, a = butter(order, normal_cutoff, btype="low", analog=False)

x_filt = filtfilt(b, a, s)   # <-- x → s

# <-------- Windowed feature extraction (RMS over time) -------->
# Window size in seconds (0.2s is a nice default for demos)
window_sec = 0.2
window_size = int(window_sec * fs)  # convert seconds -> samples

if window_size < 1:
    raise ValueError("window_size is too small; increase window_sec or fs.")

# Compute RMS in non-overlapping windows
window_rows = []
n = len(x_filt)

# If the data is shorter than one window, fall back to one window using all samples
if n < window_size:
    rms_one = float(np.sqrt(np.mean(x_filt**2)))
    window_rows.append({"window_start_s": float(t[0]), "window_end_s": float(t[-1]), "rms": rms_one})
else:
    for start in range(0, n - window_size + 1, window_size):
        end = start + window_size
        segment = x_filt[start:end]

        rms = float(np.sqrt(np.mean(segment**2)))
        window_rows.append(
            {
                "window_start_s": float(t[start]),
                "window_end_s": float(t[end - 1]),
                "rms": rms,
            }
        )

windowed_df = pd.DataFrame(window_rows)
windowed_df.to_csv("outputs/windowed_features.csv", index=False)

# <---------- Compute simple features ---------->
# Compute basic statistics on both raw and filtered signals
mean_val = float(s.mean())                 # <-- x → s
rms_val = float((s**2).mean() ** 0.5)      # <-- x → s
max_val = float(s.max())                   # <-- x → s

mean_f = float(x_filt.mean())
rms_f = float((x_filt**2).mean() ** 0.5)
max_f = float(x_filt.max())

features = pd.DataFrame(
    [
        {"signal": "raw", "mean": mean_val, "rms": rms_val, "max": max_val},
        {"signal": "filtered", "mean": mean_f, "rms": rms_f, "max": max_f},
    ]
)

features.to_csv("outputs/features.csv", index=False)

# <---- Plotting ---->
plt.plot(t, s, label="raw")        # <-- x → s
plt.plot(t, x_filt, label="filtered")
plt.xlabel("Time (seconds)")
plt.ylabel("Signal value")
plt.title("Raw vs Filtered Sensor Signal")
plt.legend()

# Save the plot
plt.savefig("outputs/raw_vs_filtered.png", dpi=200, bbox_inches="tight")
print("Saved outputs/raw_vs_filtered.png and outputs/features.csv")

# <------ Plot RMS over time ------->
# Use the midpoint of each window for the x-axis
midpoints = (windowed_df["window_start_s"] + windowed_df["window_end_s"]) / 2

plt.figure()
plt.plot(midpoints, windowed_df["rms"])
plt.xlabel("Time (seconds)")
plt.ylabel("RMS (filtered signal)")
plt.title("Windowed RMS Over Time")
plt.savefig("outputs/rms_over_time.png", dpi=200, bbox_inches="tight")
print("Saved outputs/rms_over_time.png and outputs/windowed_features.csv")