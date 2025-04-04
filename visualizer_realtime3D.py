import matplotlib.pyplot as plt
import numpy as np
import threading
from collections import deque
from matplotlib.animation import FuncAnimation
import mne
from mne.viz import Brain
import logging
import time
import random


import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class RealtimeEEGVisualizer:
    logger = logging.getLogger(__name__)

    def __init__(self, buffer_size=1000, num_channels=14):
        self.buffer_size = buffer_size
        self.num_channels = num_channels
        self.data_buffers = [deque(maxlen=buffer_size) for _ in range(num_channels)]
        self.gyro_x_buffer = deque(maxlen=buffer_size)
        self.gyro_y_buffer = deque(maxlen=buffer_size)

        # EEG channel names for Emotiv EPOC+
        self.channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2',
                              'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

        # 3D coordinates for the EEG electrodes (you can modify these coordinates)
        self.eeg_positions = {
            'AF3': (-0.3, 0.6, 0.8), 'F7': (-0.5, 0.4, 0.7), 'F3': (-0.6, 0.2, 0.7),
            'FC5': (-0.7, 0.0, 0.6), 'T7': (-0.8, -0.2, 0.5), 'P7': (-0.9, -0.4, 0.4),
            'O1': (-1.0, -0.5, 0.2), 'O2': (1.0, -0.5, 0.2), 'P8': (0.9, -0.4, 0.4),
            'T8': (0.8, -0.2, 0.5), 'FC6': (0.7, 0.0, 0.6), 'F4': (0.6, 0.2, 0.7),
            'F8': (0.5, 0.4, 0.7), 'AF4': (0.3, 0.6, 0.8)
        }
        self.channel_to_label = {
        'AF3': 'lh.BA44.label',
        'F7': 'lh.BA45.label',
        'F3': 'lh.BA6.label',
        'FC5': 'lh.BA4a.label',
        'T7': 'lh.BA4p.label',
        'P7': 'lh.BA3b.label',
        'O1': 'lh.V1.label',
        'O2': 'rh.V1.label',
        'P8': 'rh.BA3b.label',
        'T8': 'rh.BA4p.label',
        'FC6': 'rh.BA4a.label',
        'F4': 'rh.BA6.label',
        'F8': 'rh.BA45.label',
        'AF4': 'rh.BA44.label'
    }
        plt.style.use('dark_background')

    # Initialize 3D Brain Visualization
        self.subject = 'sample'  # Use MNE sample dataset for illustration
        self.data_path = mne.datasets.sample.data_path()
        self.subjects_dir = str(self.data_path) + '/subjects'
        self.sample_dir = self.data_path / "MEG" / "sample"
        brain_kwargs = dict(alpha=0.5, background="white", cortex="low_contrast")


        self.label_vertices = {}
        for channel, label_name in self.channel_to_label.items():
            label_path = f"{self.subjects_dir}/fsaverage/label/{label_name}"
            self.label_vertices[channel] = mne.read_label(label_path).vertices

        # Initialize 2D Figure
        self.fig_2d, axes_2d = plt.subplots(15, 1, figsize=(18, 12), gridspec_kw={'height_ratios': [1] * 14 + [2]})
        self.ax_eeg = axes_2d[:14]  # 14 EEG Subplots
        self.ax_gyro = axes_2d[14]  # Gyro 2D Motion Plot

        # EEG Signal Plot (Each Channel in Separate Row)
        colors = plt.cm.viridis(np.linspace(0, 1, num_channels))
        self.lines = [self.ax_eeg[i].plot([], [], label=self.channel_names[i], color=colors[i])[0]
                      for i in range(self.num_channels)]

        # Format EEG subplots
        for i, ax in enumerate(self.ax_eeg):
            ax.set_ylabel(self.channel_names[i])
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.4)
            ax.set_ylim(-500, 500)

        # Only the top EEG subplot gets the title
        self.ax_eeg[0].set_title("Real-time EEG Signals (14 Channels)", fontsize=14)

        # Gyro 2D Trajectory Plot (Head Movement)
        self.scatter_gyro, = self.ax_gyro.plot([], [], 'wo', markersize=4, alpha=0.6, label="Head Movement")  # Scatter plot
        self.line_gyro_traj, = self.ax_gyro.plot([], [], 'c-', linewidth=1, alpha=0.8, label="Trajectory")  # Trajectory line
        self.ax_gyro.set_title("Real-time Head Movement (Gyro X vs Gyro Y)", fontsize=14)
        self.ax_gyro.set_xlabel("Gyro X (Left-Right)")
        self.ax_gyro.set_ylabel("Gyro Y (Up-Down)")
        self.ax_gyro.set_xlim(0, 150)
        self.ax_gyro.set_ylim(0, 150)
        self.ax_gyro.legend(loc='upper right')
        self.ax_gyro.grid(True, alpha=0.4)
        plt.subplots_adjust(hspace=0.5)  # Adjust the space between plots (increase the value for more space)


        # Create the MNE Brain object
        #self.brain = Brain(self.subject, hemi='both', surf='pial', subjects_dir=self.subjects_dir, size=(800, 600), background='white')
        self.brain = Brain("sample", hemi="both", surf="pial", subjects_dir=self.subjects_dir, **brain_kwargs)
        self.brain.show_view(azimuth=190, elevation=70, distance=350, focalpoint=(0, 0, 20))
        # Add labels for the 14 EEG channels
        for channel, label_name in self.channel_to_label.items():
            hemi = "lh" if "lh" in label_name else "rh"
            color = (random.random(), random.random(), random.random())  # RGB color

            self.brain.add_label(label_name.split('.')[1], hemi=hemi, color=color, borders=True)
        # Add EEG electrodes (from self.eeg_positions) as custom markers on the brain surface
        #for channel_name, position in self.eeg_positions.items():
         #   self.brain.add_marker(position, color='red', scale=0.5)

    def update_2d(self, frame):
        """
        Update the 2D plots with the latest data.
        """
        # Update EEG data for each channel separately
        for i, line in enumerate(self.lines):
            if len(self.data_buffers[i]) > 0:
                x_data = list(range(len(self.data_buffers[i])))
                y_data = list(self.data_buffers[i])
                line.set_data(x_data, y_data)
                ax = self.ax_eeg[i]
                ax.relim()
                ax.autoscale_view()

        # Update Gyro 2D Trajectory Plot
        if len(self.gyro_x_buffer) > 1:
            #self.logger.info(f"Updating 2D plot with Gyro data: gyro_x_buffer={list(self.gyro_x_buffer)}, gyro_y_buffer={list(self.gyro_y_buffer)}")
            self.scatter_gyro.set_data(self.gyro_x_buffer, self.gyro_y_buffer)  # Update scatter plot
            self.line_gyro_traj.set_data(self.gyro_x_buffer, self.gyro_y_buffer)  # Update trajectory line
            self.ax_gyro.relim()
            self.ax_gyro.autoscale_view()

        return self.lines + [self.scatter_gyro, self.line_gyro_traj]


    def update_3d(self, frame):
        # Initialize data for all vertices
        lh_data = np.zeros(len(self.brain.geo['lh'].coords))
        rh_data = np.zeros(len(self.brain.geo['rh'].coords))

        # Map EEG data to the corresponding label vertices
        for i, channel in enumerate(self.channel_names):
            if len(self.data_buffers[i]) > 0:
                avg_data = np.mean(list(self.data_buffers[i]))
                vertices = self.label_vertices[channel]
                if 'lh' in self.channel_to_label[channel]:
                    lh_data[vertices] = avg_data
                elif 'rh' in self.channel_to_label[channel]:
                    rh_data[vertices] = avg_data

        # Clear existing labels to refresh them
        self.brain.remove_labels()

        # Update labels with random colors
        for channel, label_name in self.channel_to_label.items():
            hemi = "lh" if "lh" in label_name else "rh"
            color = (random.random(), random.random(), random.random())  # RGB color
            self.brain.add_label(label_name.split('.')[1], hemi=hemi, color=color, borders=True)

        # Add the data to the brain visualization
        self.brain.add_data(
            lh_data,
            hemi="lh",
            fmin=-500,
            fmax=500,
            colormap="viridis",
            alpha=0.8,
            smoothing_steps=10,
        )
        self.brain.add_data(
            rh_data,
            hemi="rh",
            fmin=-500,
            fmax=500,
            colormap="viridis",
            alpha=0.4,
            smoothing_steps=10,
        )

        # Force the Brain object to refresh its visualization
        self.brain.show_view(azimuth=190, elevation=70, distance=350, focalpoint=(0, 0, 20))
        return []


    def update(self, frame):
            # Update EEG data for each channel separately
            for i, line in enumerate(self.lines):
                if len(self.data_buffers[i]) > 0:
                    x_data = list(range(len(self.data_buffers[i])))
                    y_data = list(self.data_buffers[i])
                    line.set_data(x_data, y_data)
                    ax = self.ax_eeg[i]
                    ax.relim()
                    ax.autoscale_view()

            # Update Gyro 2D Trajectory Plot
            if len(self.gyro_x_buffer) > 1:
                self.scatter_gyro.set_data(self.gyro_x_buffer, self.gyro_y_buffer)  # Update scatter plot
                self.line_gyro_traj.set_data(self.gyro_x_buffer, self.gyro_y_buffer)  # Update trajectory line
                self.ax_gyro.relim()
                self.ax_gyro.autoscale_view()

            updated_lines = self.lines + [self.scatter_gyro, self.line_gyro_traj]
            return updated_lines

    def update_gyro_data(self, gyro_x, gyro_y):
        """
        Update the Gyro buffers with new data.
        """
        #self.logger.info(f"Updating Gyro buffers: gyro_x={gyro_x}, gyro_y={gyro_y}")
        self.gyro_x_buffer.append(gyro_x)
        self.gyro_y_buffer.append(gyro_y)
        #self.logger.info(f"Gyro buffers updated: gyro_x_buffer={list(self.gyro_x_buffer)}, gyro_y_buffer={list(self.gyro_y_buffer)}")

    def start_visualization(self, type='2d'):
        if type == '2d':
            # Start 2D visualization using matplotlib animation
            anim_2d = FuncAnimation(self.fig_2d, self.update_2d, interval=100)
            plt.show()
        elif type == '3d':
            # For 3D visualization, the Brain object handles rendering
            self.logger.info("Starting 3D brain visualization...")
            while True:
                try:
                    self.update_3d(None)  # Update the 3D brain visualization
                    time.sleep(0.1)  # Add a small delay to prevent overwhelming the system
                except KeyboardInterrupt:
                    self.logger.info("3D visualization interrupted by user.")
                    break



    def run_visualizations(self):
        # Start the 2D visualization in a separate thread
        thread_2d = threading.Thread(target=self.start_visualization, args=('2d',))
        thread_2d.start()

        # Start the 3D visualization in a separate thread
        thread_3d = threading.Thread(target=self.start_visualization, args=('3d',))
        thread_3d.start()

    def add_data(self, data):
        """
        Add raw EEG data to the buffers for visualization.
        """
        if isinstance(data, dict):  # Assuming data is a dictionary with channel names as keys
            for i, channel in enumerate(self.channel_names):
                if channel in data:
                    self.data_buffers[i].append(data[channel])
        elif isinstance(data, list):  # Handle raw data as a list
            for i in range(min(len(data), self.num_channels)):
                self.data_buffers[i].append(data[i])
        else:
            self.logger.warning("Unsupported data format for visualization.")
