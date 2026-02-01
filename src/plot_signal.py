import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
data = pd.read_csv("data/sample_signal.csv")

# Plot the signal
plt.plot(data["time"], data["signal"])

# Labels and title
plt.xlabel("Time (seconds)")
plt.ylabel("Signal value")
plt.title("Sensor Signal Over Time")

# Show the plot
plt.savefig("outputs/signal_plot.png", dpi=200, bbox_inches="tight")
print("Saved plot to outputs/signal_plot.png")
