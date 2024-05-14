"""Get relevant imaging settings

- single plane vs. fast-z mode
- Frame averaging

# ThorSync
"""
from pathlib import Path
import utils2p
from attrs import define, field
import cattrs
import pprint as pp
import yaml


@define
class FunctionalImagingSettings:
    raw_frame_rate: float
    frame_rate: float
    n_frames: int
    n_timepoints: int
    fast_z: bool
    n_z: int
    n_flyback_frames: int
    step_size_um: float
    average_mode: bool
    n_averaging: int
    steps_per_frame: int
    timelapse_trigger_mode: int
    streaming_trigger_mode: int
    stimulus_triggering: int
    software_version: str
    experiment_notes: str
    thorsync_name: str = field(default=None)

    @classmethod
    def from_xml(cls, xml_file):
        meta = utils2p.Metadata(xml_file)

        raw_frame_rate = float(meta.get_metadata_value("LSM", "frameRate"))
        n_frames = int(meta.get_metadata_value("Streaming", "frames"))
        fast_z = bool(int(meta.get_metadata_value("Streaming", 'zFastEnable')))

        software_version = meta.get_metadata_value("Software", 'version')

        timelapse_trigger_mode = int(meta.get_metadata_value("Timelapse", 'triggerMode'))
        streaming_trigger_mode = int(meta.get_metadata_value("Streaming", 'triggerMode'))
        stimulus_triggering = int(meta.get_metadata_value("Streaming", 'stimulusTriggering'))

        experiment_notes = meta.get_metadata_value("ExperimentNotes", 'text')

        # volumetric vs. single plane
        if fast_z:
            n_z = meta.get_n_z()
            n_flyback_frames = meta.get_n_flyback_frames()
            step_size_um = meta.get_z_step_size()
            average_mode = False
        else:
            n_z = 1
            n_flyback_frames = 0
            step_size_um = 0.0
            average_mode = bool(int(meta.get_metadata_value('LSM', 'averageMode')))

        if average_mode:
            n_averaging = meta.get_n_averaging()
        else:
            n_averaging = 1

        # compute steps per plane (how to downsample the frame counter timestamps)
        if fast_z:
            steps_per_frame = n_z + n_flyback_frames
        else:  # if single plane
            steps_per_frame = n_averaging

        frame_rate = raw_frame_rate / steps_per_frame
        return cls(raw_frame_rate=raw_frame_rate,
                   frame_rate=frame_rate,
                   n_frames=n_frames,
                   n_timepoints=meta.get_n_time_points(),
                   fast_z=fast_z,
                   n_z=n_z,
                   n_flyback_frames=n_flyback_frames,
                   step_size_um=step_size_um,
                   average_mode=average_mode,
                   n_averaging=n_averaging,
                   steps_per_frame=steps_per_frame,
                   timelapse_trigger_mode=timelapse_trigger_mode,
                   streaming_trigger_mode=streaming_trigger_mode,
                   stimulus_triggering=stimulus_triggering,
                   software_version=software_version,
                   experiment_notes=experiment_notes
                   )

    @classmethod
    def from_yaml(cls, file):
        with open(file, 'r') as stream:
            yaml_settings = yaml.safe_load(stream)
        return cattrs.structure(yaml_settings, cls)

    def to_dict(self):
        return cattrs.unstructure(self)


# %%
if __name__ == '__main__':
    # %%
    LEXI_RAW_DIR = Path("/home/remy/HongLab @ Caltech Dropbox/Remy/for_lexi/raw_data")
    LEXI_PROC_DIR = Path("/local/matrix/Lexi/rotation_project/processed_data")

    date_imaged = '2024-05-09'
    fly_num = 1
    thorimage_name = 'lexi10_fastz'

    thorimage_xml = LEXI_RAW_DIR.joinpath(date_imaged,
                                          str(fly_num),
                                          thorimage_name,
                                          'Experiment.xml')
    # %%
    thorimage_settings = FunctionalImagingSettings.from_xml(thorimage_xml)
    # %%
    print('\n--------------------------------------------------')
    print(thorimage_xml.relative_to(LEXI_RAW_DIR))
    print('--------------------------------------------------')

    pp.pprint(thorimage_settings.to_dict(), sort_dicts=False)
    # %%
    mov_dir = LEXI_PROC_DIR.joinpath(thorimage_xml.relative_to(LEXI_RAW_DIR)).parent

    yaml_file = mov_dir.joinpath('thorimage_settings.yaml')
    with open(yaml_file, 'w') as f:
        yaml.dump(thorimage_settings.to_dict(), f, sort_keys=False)
    # %%
