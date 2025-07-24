# #!/usr/bin/env python3
# """
# Live lane line visualization using matplotlib
# This script connects to the model output and displays lane lines in real-time
# """

# import os
# import time
# import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')  # Use interactive backend for live display
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import cereal.messaging as messaging
# from cereal import log
# from openpilot.selfdrive.modeld.constants import ModelConstants
# from openpilot.common.swaglog import cloudlog

# class LaneLineVisualizer:
#     def __init__(self, save_plots=False, output_dir="lane_plots"):
#         self.save_plots = save_plots
#         self.output_dir = output_dir
#         if save_plots:
#             os.makedirs(output_dir, exist_ok=True)

#         # Create interactive figure
#         self.fig, self.ax = plt.subplots(figsize=(12, 8))
#         self.fig.suptitle('Live Lane Line Detection', fontsize=16)
#         self.ax.set_xlim(-50, 50)
#         self.ax.set_ylim(0, 100)
#         self.ax.set_xlabel('Lateral Distance (m)')
#         self.ax.set_ylabel('Forward Distance (m)')
#         self.ax.grid(True, alpha=0.3)

#         self.lane_colors = ['blue', 'green', 'red', 'orange']
#         self.lane_labels = ['Left Edge', 'Left Lane', 'Right Lane', 'Right Edge']
#         self.lane_plots = [self.ax.plot([], [], color=color, linewidth=2, label=label, alpha=0.8)[0]
#                            for color, label in zip(self.lane_colors, self.lane_labels)]
#         self.road_edge_plots = [self.ax.plot([], [], color='gray', linestyle='--', linewidth=1, alpha=0.5, label=f'Road Edge {i+1}')[0] for i in range(2)]
#         self.vehicle_marker = self.ax.scatter([0], [0], color='black', s=100, marker='s', zorder=10, label='Vehicle')
#         self.ax.legend()

#         self.lane_data = {i: {'x': [], 'y': []} for i in range(4)}
#         self.road_edge_data = {i: {'x': [], 'y': []} for i in range(2)}
#         self.sm = messaging.SubMaster(['modelV2'])
#         self.texts = []
#         self.frame_count = 0

#         # Add status text
#         self.status_text = self.ax.text(0.02, 0.98, 'Status: Starting...',
#                                        transform=self.ax.transAxes,
#                                        verticalalignment='top',
#                                        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))

#     def update_lane_lines(self, model_output):
#         lane_lines = model_output.laneLines
#         if lane_lines is None:
#             return
#         try:
#             lane_lines_list = list(lane_lines)
#         except Exception as e:
#             cloudlog.error(f"Error converting lane lines to list: {e}")
#             return
#         for i in range(4):
#             self.lane_data[i]['x'] = []
#             self.lane_data[i]['y'] = []
#         for i, lane_line in enumerate(lane_lines_list):
#             if i >= 4:
#                 break
#             if hasattr(lane_line, 'x') and hasattr(lane_line, 'y'):
#                 try:
#                     x_coords = list(lane_line.x)
#                     y_coords = list(lane_line.y)
#                 except Exception as e:
#                     cloudlog.error(f"Error converting lane line coordinates to list: {e}")
#                     continue
#                 if len(x_coords) == 0 or len(y_coords) == 0:
#                     continue
#                 x_coords = np.array([float(x) for x in x_coords])
#                 y_coords = np.array([float(y) for y in y_coords])
#                 valid_mask = np.isfinite(x_coords) & np.isfinite(y_coords)
#                 x_valid = x_coords[valid_mask]
#                 y_valid = y_coords[valid_mask]
#                 self.lane_data[i]['x'] = x_valid.tolist()
#                 self.lane_data[i]['y'] = y_valid.tolist()
#                 self.lane_plots[i].set_data(x_valid, y_valid)
#             else:
#                 self.lane_plots[i].set_data([], [])

#     def update_road_edges(self, model_output):
#         road_edges = model_output.roadEdges
#         if road_edges is None:
#             return
#         try:
#             road_edges_list = list(road_edges)
#         except Exception as e:
#             cloudlog.error(f"Error converting road edges to list: {e}")
#             return
#         for i in range(2):
#             self.road_edge_data[i]['x'] = []
#             self.road_edge_data[i]['y'] = []
#         for i, edge in enumerate(road_edges_list):
#             if i >= 2:
#                 break
#             if hasattr(edge, 'x') and hasattr(edge, 'y'):
#                 try:
#                     x_coords = list(edge.x)
#                     y_coords = list(edge.y)
#                 except Exception as e:
#                     cloudlog.error(f"Error converting road edge coordinates to list: {e}")
#                     continue
#                 if len(x_coords) == 0 or len(y_coords) == 0:
#                     continue
#                 x_coords = np.array([float(x) for x in x_coords])
#                 y_coords = np.array([float(y) for y in y_coords])
#                 valid_mask = np.isfinite(x_coords) & np.isfinite(y_coords)
#                 x_valid = x_coords[valid_mask]
#                 y_valid = y_coords[valid_mask]
#                 self.road_edge_data[i]['x'] = x_valid.tolist()
#                 self.road_edge_data[i]['y'] = y_valid.tolist()
#                 self.road_edge_plots[i].set_data(x_valid, y_valid)
#             else:
#                 self.road_edge_plots[i].set_data([], [])

#     def update_lane_probabilities(self, model_output):
#         lane_probs = model_output.laneLineProbs
#         if lane_probs is None:
#             return
#         try:
#             lane_probs_list = list(lane_probs)
#         except Exception as e:
#             cloudlog.error(f"Error converting lane probabilities to list: {e}")
#             return
#         # Remove previous texts
#         for txt in self.texts:
#             txt.remove()
#         self.texts = []
#         for i, prob in enumerate(lane_probs_list):
#             if i < 4 and len(self.lane_data[i]['x']) > 0:
#                 x_pos = float(np.mean(self.lane_data[i]['x']))
#                 y_pos = float(np.mean(self.lane_data[i]['y']))
#                 self.texts.append(
#                     self.ax.text(x_pos, y_pos, f'{float(prob):.2f}',
#                         color=self.lane_colors[i], fontsize=10,
#                         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7)))

#     def save_plot(self):
#         """Save the current plot to a file (optional)"""
#         if self.save_plots:
#             filename = os.path.join(self.output_dir, f"lane_plot_{self.frame_count:06d}.png")
#             self.fig.savefig(filename, dpi=150, bbox_inches='tight')
#             print(f"Saved plot: {filename}")
#         self.frame_count += 1

#     def update_visualization(self, model_output):
#         self.update_lane_lines(model_output)
#         self.update_road_edges(model_output)
#         self.update_lane_probabilities(model_output)

#         # Update status
#         self.status_text.set_text(f'Status: Live | Frame: {self.frame_count}')

#         # Optionally save plot
#         if self.save_plots:
#             self.save_plot()
#         else:
#             self.frame_count += 1

#     def animate(self, frame):
#         """Animation function for matplotlib FuncAnimation"""
#         self.sm.update(timeout=10)  # Shorter timeout for smoother animation
#         if self.sm.updated['modelV2']:
#             model_output = self.sm['modelV2']
#             self.update_visualization(model_output)

#         # Return all artists that need to be redrawn
#         return self.lane_plots + self.road_edge_plots + [self.vehicle_marker] + self.texts + [self.status_text]

#     def run_visualization(self):
#         cloudlog.info("Starting live lane line visualization...")
#         print("DEBUG: Starting live visualization...")
#         print("Close the matplotlib window to stop the visualization.")

#         try:
#             # Create animation
#             ani = animation.FuncAnimation(self.fig, self.animate, interval=50, blit=False, cache_frame_data=False)

#             # Show the plot
#             plt.show()

#         except KeyboardInterrupt:
#             cloudlog.info("Stopping lane line visualization...")
#         except Exception as e:
#             cloudlog.error(f"Error in visualization loop: {e}")
#         finally:
#             plt.close()

# def main():
#     print("Live Lane Line Visualization")
#     print("=" * 30)
#     print("This script will display lane lines in real-time.")
#     print("Close the matplotlib window to stop.")
#     print()

#     # Ask user if they want to save plots too
#     save_plots = input("Do you want to save plots to files as well? (y/n): ").lower().startswith('y')

#     visualizer = LaneLineVisualizer(save_plots=save_plots)
#     visualizer.run_visualization()

# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
"""
Live Lane Line Visualization using PyQtGraph
This script subscribes to the OpenPilot 'modelV2' stream and renders lane lines and road edges in real-time.
"""

import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import cereal.messaging as messaging

class LiveLaneLineViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Lane Line Visualization")
        self.resize(1000, 700)

        # Create the plot widget
        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)
        self.plot_widget.setXRange(-20, 20)
        self.plot_widget.setYRange(0, 20  0)
        self.plot_widget.setLabel('left', 'Forward Distance (m)')
        self.plot_widget.setLabel('bottom', 'Lateral Distance (m)')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # Colors for lanes: Left Edge, Left Lane, Right Lane, Right Edge
        self.lane_colors = ['#0066FF', '#00CC00', '#FF3300', '#FF9900']
        self.lane_curves = [self.plot_widget.plot(pen=pg.mkPen(color, width=2)) for color in self.lane_colors]

        # Road edges (gray dashed)
        self.road_edge_curves = [self.plot_widget.plot(pen=pg.mkPen('gray', width=1, style=QtCore.Qt.DashLine))
                                 for _ in range(2)]

        # Vehicle marker
        self.vehicle_marker = self.plot_widget.plot([0], [0], pen=None, symbol='s', symbolBrush='black', symbolSize=12)

        # Setup SubMaster to listen to 'modelV2'
        self.sm = messaging.SubMaster(['modelV2'])

        # Timer for updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)

    def update_plot(self):
        self.sm.update(0)
        if self.sm.updated['modelV2']:
            model = self.sm['modelV2']

            # Lane lines
            for i in range(4):
                try:
                    x = np.array(model.laneLines[i].x)
                    y = np.array(model.laneLines[i].y)
                    if len(x) and len(y):
                        self.lane_curves[i].setData(y, x)
                    else:
                        self.lane_curves[i].setData([], [])
                except Exception:
                    self.lane_curves[i].setData([], [])

            # Road edges
            for i in range(2):
                try:
                    x = np.array(model.roadEdges[i].y)
                    y = np.array(model.roadEdges[i].x)
                    if len(x) and len(y):
                        self.road_edge_curves[i].setData(x, y)
                    else:
                        self.road_edge_curves[i].setData([], [])
                except Exception:
                    self.road_edge_curves[i].setData([], [])


def main():
    app = QtWidgets.QApplication(sys.argv)
    viewer = LiveLaneLineViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
