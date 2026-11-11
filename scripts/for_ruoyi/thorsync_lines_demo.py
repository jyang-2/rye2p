"""
First stage of processing:

"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import signal
import utils2p.synchronization
import yaml


@dataclass
class ExperimentConfig:
    microscope: str
    date_imaged: str
    fly_num: int
    thorimage_name: str
    thorsync_name: str
    olf_config: str

    # Path configuration
    project_data_dir: Path = Path("/local/tensor/Ruoyi/ruoyi_rotation_project")
    raw_data_dir: Path = project_data_dir / "raw_data"
    processed_data_dir: Path = project_data_dir / "processed_data"
    olf_config_dir: Path = project_data_dir / "olfactometer_configs"

    def __post_init__(self):
        self.thorimage_folder = self.raw_data_dir / self.date_imaged / str(
            self.fly_num) / self.thorimage_name
        self.thorsync_folder = self.raw_data_dir / self.date_imaged / str(
            self.fly_num) / self.thorsync_name
        self.proc_mov_dir = self.processed_data_dir / self.date_imaged / str(
            self.fly_num) / self.thorimage_name


class SyncLineProcessor:
    TARGET_SAMPLING_RATE = 1000
    VOLTAGE_THRESHOLD = 0.8

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.sync_lines = None
        self.sync_lines_ds = None
        self.proc_sync_lines = None
        self.sync_meta = None
        self.sync_times = None

    def load_thorsync_line_names(self) -> Dict:
        """Load ThorSync line name mapping config"""
        line_config_folder = Path("/home/remy/PycharmProjects/rye2p/data/microscopes")
        thorsync_line_file = line_config_folder / f"{self.config.microscope}/thorsync_lines.yaml"

        with open(thorsync_line_file, 'r') as f:
            return yaml.safe_load(f)

    def process_sync_data(self):
        """Process and downsample sync data"""
        line_mapping = self.load_thorsync_line_names()

        # Load lines from ThorSync h5 file
        self.sync_lines = utils2p.synchronization.get_lines_from_sync_file(
                self.config.thorsync_folder / 'Episode001.h5',
                line_mapping.values()
                )
        self.sync_lines = dict(zip(line_mapping.keys(), self.sync_lines))

        # Load metadata and generate timestamps
        self.sync_meta = utils2p.synchronization.SyncMetadata(
                self.config.thorsync_folder / "ThorRealTimeDataSettings.xml"
                )
        self.sync_times = utils2p.synchronization.get_times(
                self.sync_lines['frame_counter'].size,
                self.sync_meta.get_freq()
                )

        # Downsample to target frequency
        n_decimate = int(self.sync_meta.get_freq() / self.TARGET_SAMPLING_RATE)
        self.sync_lines_ds = {
            k: signal.decimate(v, q=n_decimate)
            for k, v in self.sync_lines.items()
            if k != 'frame_counter'
            }
        self.sync_lines_ds['frame_counter'] = self.sync_lines['frame_counter'][::n_decimate]
        self.sync_lines_ds['time'] = self.sync_times[::n_decimate]

        # Process TTL pulses and monitoring signals
        self._process_sync_lines()

        # Save processed data
        np.savez_compressed(
                self.config.proc_mov_dir / 'sync_lines.npz',
                **self.sync_lines_ds
                )

    def _process_sync_lines(self):
        """Scale TTL pulse lines and include monitoring signals"""
        # Scale TTL lines between 0 and 1
        scope_pin = self.sync_lines_ds['scope_pin'] / self.sync_lines_ds['scope_pin'].max()
        olf_disp_pin = self.sync_lines_ds['olf_disp_pin'] / self.sync_lines_ds['olf_disp_pin'].max()
        lightpath_shutter = self.sync_lines_ds['lightpath_shutter'] / self.sync_lines_ds[
            'lightpath_shutter'].max()

        # Include monitoring signals
        piezo_monitor = self.sync_lines_ds['piezo_monitor']  # Don't normalize monitoring signals
        frame_counter = self.sync_lines_ds['frame_counter']

        # Store processed lines
        self.proc_sync_lines = {
            # TTL signals (normalized)
            'scope_pin': scope_pin,
            'olf_disp_pin': olf_disp_pin,
            'lightpath_shutter': lightpath_shutter,
            # Monitoring signals (raw)
            'piezo_monitor': piezo_monitor,
            'frame_counter': frame_counter,
            # Time
            'time': self.sync_lines_ds['time']
            }

    def extract_named_line_pulse_times(self, line_name: str,
                                       voltage_threshold: Optional[float] = None) -> Tuple[
        np.ndarray, np.ndarray]:
        """Extract pulse start and end times from specified signal line
    
        Args:
            line_name: Name of the sync line to process
            voltage_threshold: Optional threshold value for pulse detection. If None, uses class 
            default
    
        Returns:
            Tuple of (start_times, end_times) arrays
        """
        if self.proc_sync_lines is None:
            raise RuntimeError("Process sync data first by calling process_sync_data()")

        if line_name not in self.proc_sync_lines:
            raise ValueError(f"Line '{line_name}' not found in processed sync lines")

        line_signal = self.proc_sync_lines[line_name]
        threshold = voltage_threshold if voltage_threshold is not None else self.VOLTAGE_THRESHOLD

        start_idx, = utils2p.synchronization.edges(line_signal, size=threshold)
        end_idx, = utils2p.synchronization.edges(
                line_signal,
                size=(-np.inf, -1 * threshold)
                )

        return (
            self.proc_sync_lines['time'][start_idx],
            self.proc_sync_lines['time'][end_idx]
            )

    def extract_pulse_times(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pulse start and end times from scope pin signal"""
        return self.extract_named_line_pulse_times('scope_pin')

    def get_monitoring_signals(self) -> Dict[str, np.ndarray]:
        """Get monitoring signals like piezo and frame counter"""
        if self.proc_sync_lines is None:
            raise RuntimeError("Process sync data first by calling process_sync_data()")

        return {
            'piezo_monitor': self.proc_sync_lines['piezo_monitor'],
            'frame_counter': self.proc_sync_lines['frame_counter'],
            'time': self.proc_sync_lines['time']
            }
# %%
from attrs import define, field


@define
class StackTimeCorrector:
    n_frames: int
    n_timepoints: int
    steps_per_frame: int
    fast_z: bool
    average_mode: bool
    n_averaging: int


    def correct_times(self, stack_times: np.ndarray, frame_times: np.ndarray):
        """Correct stack and frame times using scope pulse data

        Args:
            stack_times: Original stack timing data
            frame_times: Original frame timing data
        """
        if self.sync_processor.proc_sync_lines is None:

        raise RuntimeError("Process sync data first using process_sync_data()")

        pulse_starts, pulse_ends = self.sync_processor.extract_pulse_times()

        self.corrected_frame_times = utils2p.fix_frame_times.frames_by_scope_pulse(
                frame_times,
                pulse_starts,
                pulse_ends
                )

        # Correct stack times by taking the first frame time of each stack
        frames_per_stack = len(frame_times) // len(stack_times)
        self.corrected_stack_times = self.corrected_frame_times[::frames_per_stack]

    def get_corrected_times(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the corrected stack and frame times

        Returns:
Tuple of (corrected_stack_times, corrected_frame_times)
        """
        if self.corrected_stack_times is None or self.corrected_frame_times is None:

    raise RuntimeError("Call correct_times first to generate corrected timing data")

    return self.corrected_stack_times, self.corrected_frame_times

