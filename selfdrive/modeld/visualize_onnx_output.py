#!/usr/bin/env python3
"""
Visualize ONNX model output for lane line detection
This script can be used to visualize the raw ONNX model outputs
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
import time
import argparse
from pathlib import Path

# Set up matplotlib for real-time plotting
plt.ion()  # Turn on interactive mode

class ONNXLaneVisualizer:
    def __init__(self, model_output_size=1536):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.suptitle('ONNX Model Lane Line Output', fontsize=16)

        # Set up the plot
        self.ax.set_xlim(-50, 50)  # Lateral distance in meters
        self.ax.set_ylim(0, 100)   # Forward distance in meters
        self.ax.set_xlabel('Lateral Distance (m)')
        self.ax.set_ylabel('Forward Distance (m)')
        self.ax.grid(True, alpha=0.3)

        # Colors for different lane lines
        self.lane_colors = ['blue', 'green', 'red', 'orange']
        self.lane_labels = ['Left Edge', 'Left Lane', 'Right Lane', 'Right Edge']

        # Initialize lane line plots
        self.lane_plots = []
        for i, (color, label) in enumerate(zip(self.lane_colors, self.lane_labels)):
            line, = self.ax.plot([], [], color=color, linewidth=2, label=label, alpha=0.8)
            self.lane_plots.append(line)

        # Add legend
        self.ax.legend()

        # Add vehicle position indicator
        self.vehicle_marker = self.ax.scatter([0], [0], color='black', s=100, marker='s', zorder=10, label='Vehicle')

        # Model output size for demo data
        self.model_output_size = model_output_size

        # Initialize data storage
        self.lane_data = {i: {'x': [], 'y': []} for i in range(4)}

    def generate_demo_lane_data(self):
        """Generate demo lane line data for visualization"""
        # Create realistic lane line data
        y_points = np.linspace(0, 100, 33)  # 33 points from 0 to 100m ahead

        # Left lane line (slightly curved)
        left_x = -1.8 + 0.01 * y_points + 0.0001 * y_points**2
        left_y = y_points

        # Right lane line (slightly curved)
        right_x = 1.8 + 0.01 * y_points + 0.0001 * y_points**2
        right_y = y_points

        # Left edge (further left)
        left_edge_x = -3.5 + 0.02 * y_points + 0.0002 * y_points**2
        left_edge_y = y_points

        # Right edge (further right)
        right_edge_x = 3.5 + 0.02 * y_points + 0.0002 * y_points**2
        right_edge_y = y_points

        # Add some noise to make it look more realistic
        noise_scale = 0.1
        left_x += np.random.normal(0, noise_scale, len(left_x))
        right_x += np.random.normal(0, noise_scale, len(right_x))
        left_edge_x += np.random.normal(0, noise_scale, len(left_edge_x))
        right_edge_x += np.random.normal(0, noise_scale, len(right_edge_x))

        return [
            np.column_stack([left_edge_x, left_edge_y]),
            np.column_stack([left_x, left_y]),
            np.column_stack([right_x, right_y]),
            np.column_stack([right_edge_x, right_edge_y])
        ]

    def update_lane_lines(self, lane_lines_data):
        """Update lane line visualization with new data"""
        try:
            # Clear previous data
            for i in range(4):
                self.lane_data[i]['x'] = []
                self.lane_data[i]['y'] = []

            # Process each lane line
            for i, lane_line in enumerate(lane_lines_data):
                if i >= 4:  # Only process first 4 lane lines
                    break

                if lane_line is not None and len(lane_line) > 0:
                    # Extract position data (x, y coordinates)
                    x_coords = lane_line[:, 0]  # Lateral position
                    y_coords = lane_line[:, 1]  # Forward position

                    # Filter out invalid points (NaN or inf)
                    valid_mask = np.isfinite(x_coords) & np.isfinite(y_coords)
                    x_valid = x_coords[valid_mask]
                    y_valid = y_coords[valid_mask]

                    # Store data
                    self.lane_data[i]['x'] = x_valid
                    self.lane_data[i]['y'] = y_valid

                    # Update plot
                    if len(x_valid) > 0:
                        self.lane_plots[i].set_data(x_valid, y_valid)
                    else:
                        self.lane_plots[i].set_data([], [])

            # Update plot limits based on data
            self.update_plot_limits()

        except Exception as e:
            print(f"Error updating lane lines: {e}")

    def update_plot_limits(self):
        """Update plot limits based on current data"""
        all_x = []
        all_y = []

        for data in self.lane_data.values():
            if len(data['x']) > 0:
                all_x.extend(data['x'])
                all_y.extend(data['y'])

        if len(all_x) > 0:
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)

            # Add some padding
            x_padding = max(5, (x_max - x_min) * 0.1)
            y_padding = max(5, (y_max - y_min) * 0.1)

            self.ax.set_xlim(x_min - x_padding, x_max + x_padding)
            self.ax.set_ylim(max(0, y_min - y_padding), y_max + y_padding)

    def add_lane_probabilities(self, lane_probs=None):
        """Add lane line probabilities as text annotations"""
        try:
            if lane_probs is None:
                # Generate demo probabilities
                lane_probs = [0.95, 0.98, 0.97, 0.92]

            # Clear previous text annotations
            for txt in self.ax.texts:
                txt.remove()

            # Add probability text for each lane
            for i, prob in enumerate(lane_probs):
                if i < 4 and len(self.lane_data[i]['x']) > 0:
                    x_pos = float(np.mean(self.lane_data[i]['x']))
                    y_pos = float(np.mean(self.lane_data[i]['y']))
                    self.ax.text(x_pos, y_pos, f'{prob:.2f}',
                               color=self.lane_colors[i], fontsize=10,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        except Exception as e:
            print(f"Error adding lane probabilities: {e}")

    def update_visualization(self, lane_lines_data=None, lane_probs=None):
        """Update the complete visualization"""
        if lane_lines_data is None:
            # Generate demo data
            lane_lines_data = self.generate_demo_lane_data()

        # Clear the plot
        self.ax.clear()

        # Re-setup the plot
        self.ax.set_xlabel('Lateral Distance (m)')
        self.ax.set_ylabel('Forward Distance (m)')
        self.ax.grid(True, alpha=0.3)

        # Update lane lines
        self.update_lane_lines(lane_lines_data)

        # Add lane probabilities
        self.add_lane_probabilities(lane_probs)

        # Re-add vehicle marker
        self.vehicle_marker = self.ax.scatter([0], [0], color='black', s=100, marker='s', zorder=10, label='Vehicle')

        # Re-add legend
        self.ax.legend()

        # Update plot limits
        self.update_plot_limits()

        # Force redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def run_demo(self, duration=30):
        """Run a demo visualization with simulated data"""
        print("Running ONNX lane line visualization demo...")
        print("Press Ctrl+C to stop.")

        start_time = time.time()

        try:
            while time.time() - start_time < duration:
                # Generate new demo data with slight variations
                lane_lines_data = self.generate_demo_lane_data()

                # Generate demo probabilities with time-varying values
                base_time = time.time()
                lane_probs = [
                    0.95 + 0.05 * np.sin(base_time),
                    0.98 + 0.02 * np.sin(base_time + 1),
                    0.97 + 0.03 * np.sin(base_time + 2),
                    0.92 + 0.08 * np.sin(base_time + 3)
                ]

                # Update visualization
                self.update_visualization(lane_lines_data, lane_probs)

                # Small delay
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("Stopping demo...")
        except Exception as e:
            print(f"Error in demo loop: {e}")
        finally:
            plt.ioff()
            plt.close()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Visualize ONNX model lane line output')
    parser.add_argument('--demo', action='store_true', help='Run demo with simulated data')
    parser.add_argument('--duration', type=int, default=30, help='Demo duration in seconds')
    parser.add_argument('--output-size', type=int, default=1536, help='Model output size')

    args = parser.parse_args()

    print("ONNX Lane Line Visualization")
    print("=" * 35)

    # Create visualizer
    visualizer = ONNXLaneVisualizer(model_output_size=args.output_size)

    if args.demo:
        visualizer.run_demo(duration=args.duration)
    else:
        print("Use --demo flag to run the visualization demo")
        print("This will show simulated lane line data for 30 seconds")

if __name__ == "__main__":
    main()