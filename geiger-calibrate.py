import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
from scipy.interpolate import interp1d

def determine_operating_voltage(file_path, start_threshold=200, end_threshold=300, interpolation_points=500):
    # Load data from the Excel file
    data = pd.read_excel(file_path)
    voltage = data['V'].values
    counts = data['C'].values

    # Interpolate data for better continuity and resolution
    interp_func = interp1d(voltage, counts, kind='cubic')
    voltage_interp = np.linspace(voltage[0], voltage[-1], interpolation_points)
    counts_interp = interp_func(voltage_interp)

    # Smooth the interpolated data
    counts_smooth = savgol_filter(counts_interp, window_length=10, polyorder=3)

    # Calculate the gradient of smoothed data
    gradient = np.gradient(counts_smooth, voltage_interp)

    # Detect the start of the plateau where the gradient falls below start_threshold
    plateau_start = next((i for i, g in enumerate(gradient) if abs(g) < start_threshold), 0)

    # Detect the end of the plateau where the gradient rises above end_threshold
    breakdown_start = next((i + plateau_start for i, g in enumerate(gradient[plateau_start:]) if abs(g) > end_threshold), len(voltage_interp) - 1)

    # Calculate the midpoint of the plateau region
    midpoint_index = (plateau_start + breakdown_start) // 2
    operating_voltage = voltage_interp[midpoint_index]

    # Plot the data with annotations
    plt.figure(figsize=(10, 6))
    plt.plot(voltage, counts, 'o', label="Original Data", alpha=0.5)
    plt.plot(voltage_interp, counts_smooth, label="Smoothed Interpolated Data")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Count (C)")
    plt.title("Geiger Counter Voltage-Count Curve with Smoothing")
    plt.grid(True)

    # Mark regions on the plot
    plt.axvline(voltage_interp[plateau_start], color='orange', linestyle='--', label="Plateau Start")
    plt.axvline(voltage_interp[breakdown_start], color='red', linestyle='--', label="Breakdown Start")
    plt.axvline(operating_voltage, color='green', linestyle='--', label="Operating Voltage")
    plt.legend()

    # Display the plot
    plt.show()

    print(f"Estimated Operating Voltage: {operating_voltage:.2f} V")
    return operating_voltage

file_path = 'geiger_data.xlsx' 
operating_voltage = determine_operating_voltage(file_path, start_threshold=300, end_threshold=300)
