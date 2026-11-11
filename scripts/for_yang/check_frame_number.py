"""
1. Make thorimage_settings.yaml from the `{{ThorImage folder}}/Experiment.xml` file
2. From the ThorSync .h5 file, extract the lines
  - get and match the line names
  -
"""
import utils2p
import utils2p.synchronization
from pathlib import Path
import numpy as np
from rye2p.thorsync import linenames
import pprint as pp
from rye2p.thorimage.settings import FunctionalImagingSettings
from attrs import define, field
import shutil
import yaml
from rye2p import datadirs

save_folder = Path("/local/tensor/Remy/for_yang/check_frame_number")

mov_dir = Path("/local/tensor/Yang/For_Remy/1FLfun1")
thorimage_xml = mov_dir.joinpath("Experiment.xml")

h5_file = Path("/local/tensor/Yang/For_Remy/1FLfun1_SyncData/Episode_0000.h5")
thorsync_name = h5_file.parent.name
# %%

thorimage_settings = FunctionalImagingSettings.from_xml(thorimage_xml)
thorimage_settings.thorsync_name = thorsync_name

pp.pprint(thorimage_settings.to_dict(), sort_dicts=False)

# %%
with open(save_folder.joinpath('thorimage_settings.yaml'), 'w') as f:
    yaml.dump(thorimage_settings.to_dict(), f, sort_keys=False)
# %%

# LEXI_RAW_DIR = Path("/home/remy/HongLab @ Caltech Dropbox/Remy/for_lexi/raw_data")

LEXI_RAW_DIR = Path("/local/tensor/Remy/for_lexi/raw_data")
LEXI_PROC_DIR = Path("/local/matrix/Lexi/rotation_project/processed_data")

date_imaged = '2024-05-23'
fly_num = 2

thorimage_name = 'kiwi_ea_eb_only'
thorsync_name = 'SyncData001'

thorimage_xml = LEXI_RAW_DIR.joinpath(date_imaged,
                                      str(fly_num),
                                      thorimage_name,
                                      'Experiment.xml')
h5_file = LEXI_RAW_DIR.joinpath(date_imaged,
                                str(fly_num),
                                thorsync_name,
                                'Episode001.h5')

mov_dir = LEXI_PROC_DIR.joinpath(date_imaged,
                                 str(fly_num),
                                 thorimage_name)
# %%
thorimage_settings = FunctionalImagingSettings.from_xml(thorimage_xml)
thorimage_settings.thorsync_name = thorsync_name

print('\n--------------------------------------------------')
print(thorimage_xml.relative_to(LEXI_RAW_DIR))
print('--------------------------------------------------')

pp.pprint(thorimage_settings.to_dict(), sort_dicts=False)
# %%
# use_single_plane = True
#
# if use_single_plane:
#     xml_file = Path("/home/remy/HongLab @ Caltech Dropbox/Remy/for_lexi/"
#                     "raw_data/2024-04-22/2/megamat06_fast_singleplane/Experiment.xml")
#
#     h5_file = Path("/home/remy/HongLab @ Caltech Dropbox/Remy/for_lexi/"
#                    "raw_data/2024-04-22/2/SyncData003/Episode001.h5")
# else:
#     xml_file = Path("/home/remy/HongLab @ Caltech Dropbox/Remy/for_lexi/"
#                     "raw_data/2024-04-22/2/megamat06_fast/Experiment.xml")
#
#     h5_file = Path("/home/remy/HongLab @ Caltech Dropbox/Remy/for_lexi/"
#                    "raw_data/2024-04-22/2/SyncData001/Episode001.h5")

# %% Get metadata
meta = utils2p.Metadata(thorimage_xml)

sync_meta = utils2p.synchronization.SyncMetadata(h5_file.with_name("ThorRealTimeDataSettings.xml"))

# %% match the ThorSync lines

observed_line_names = lines.get_thorsync_line_names(h5_file)

matched_lines, matched_ratio = lines.get_matched_thorsync_lines(
        lines.default_line_names,
        observed_line_names)

print(f"\nMATCHED LINES:")
print('================')
pp.pprint(matched_lines, sort_dicts=False)

# # %%
#
# class ThorSyncScopePin:
#     freq: int
#     raw_scope_pin: np.array
#     min_epoch_len_in_sec: [int | float] = field()
#     scope_pin: np.array


# %% Extract lines from ThorSync .h5 file

sync_lines = utils2p.synchronization.get_lines_from_sync_file(h5_file, matched_lines.values())
sync_lines = dict(zip(matched_lines.keys(), sync_lines))

# %% Make timestamps array
sync_times = utils2p.synchronization.get_times(sync_lines['frame_counter'].size,
                                               sync_meta.get_freq()
                                               )
# %% Plot original scope pin
import matplotlib.pyplot as plt

plt.plot(sync_times, sync_lines['scope_pin'])
plt.show()
# %% Filter scope pin line
scope_pin = (sync_lines['scope_pin'] / sync_lines['scope_pin'].max()).astype(bool)

plt.plot(sync_times, scope_pin)
plt.show()
# %%
min_scope_pulse_len_in_seconds = 30
sync_lines['scope_pin'] = utils2p.synchronization.epoch_length_filter(
        scope_pin,
        min_scope_pulse_len_in_seconds * sync_meta.get_freq()
        )

# %% Get scope pin triggers

voltage_threshold = 0.5
scope_ici = utils2p.synchronization.edges(sync_lines['scope_pin'],
                                          size=voltage_threshold)
scope_fci = utils2p.synchronization.edges(sync_lines['scope_pin'],
                                          size=(-np.inf, -1 * voltage_threshold)
                                          )

scope_ict = sync_times[scope_ici]
scope_fct = sync_times[scope_fci]
# scope_fct = np.array([sync_times[-1]])

print(f"scope_ict = {scope_ict}")
print(f"scope_fct = {scope_fct}")

print(f"movie length (min): {(scope_fct - scope_ict) / 60.0}")

# %% Get olfactometer triggers

voltage_threshold = (sync_lines['olf_disp_pin'].max() - sync_lines['olf_disp_pin'].min()) / 2
voltage_threshold = voltage_threshold.round(1)

olf_ici = utils2p.synchronization.edges(sync_lines['olf_disp_pin'],
                                        size=voltage_threshold)
olf_fci = utils2p.synchronization.edges(sync_lines['olf_disp_pin'],
                                        size=(-np.inf, -1 * voltage_threshold)
                                        )

olf_ict_ori = sync_times[olf_ici]
olf_fct_ori = sync_times[olf_fci]

# olf_ict_mask = [np.bitwise_and(olf_ict_ori > scope_ict[0], olf_ict_ori < scope_fct[0])]
# olf_fct_mask = np.bitwise_and(olf_fct_ori > scope_ict[0], olf_fct_ori < scope_fct[0])
#
# print('\nOdor pulses:')
# print('-------------')
# print(f"size of `olf_ict`: {olf_ict_mask.sum()}")
# print(f"size of `olf_fct`: {olf_fct_mask.sum()}")
# %%

plt.plot(sync_times, sync_lines['scope_pin'])
ax = plt.gca()

for item in olf_ict_ori.tolist():
    ax.axvline(item, color='r', lw=0.5, zorder=0)
plt.show()
# %% Final olfactometer
# olf_ict = olf_ict_ori[olf_ict_mask]
# olf_fct = olf_fct_ori[olf_fct_mask]
olf_ict = olf_ict_ori
olf_fct = olf_fct_ori

print(olf_fct - olf_ict)
# %%
n_scope_pulse = scope_fct.size

stack_pulse_masks = [
    np.bitwise_and(raw_stack_times > scope_ict[i], raw_stack_times < scope_fct[i])
    for i in range(n_scope_pulse)
    ]
# %% Trim the frame times for each scope pulse

from rye2p.fix_frame_times import frames_by_scope_pulse

frame_times_by_pulse = frames_by_scope_pulse(raw_frame_times, scope_ict, scope_fct)

print([item.size for item in frame_times_by_pulse])

stack_times_by_pulse = [item[::3] for item in frame_times_by_pulse]

# %%
raw_frame_times = utils2p.synchronization.get_start_times(sync_lines['frame_counter'],
                                                          sync_times,
                                                          zero_based_counter=False)
masked_raw_frame_times = raw_frame_times[raw_frame_times > scope_ict[0]]

masked_raw_frame_times_concat = np.concatenate(stack_times_by_pulse)

# %% Stack times

processed_frame_counter = utils2p.synchronization.process_frame_counter(
        sync_lines['frame_counter'],
        steps_per_frame=thorimage_settings.steps_per_frame)

raw_stack_times = utils2p.synchronization.get_start_times(processed_frame_counter,
                                                          sync_times,
                                                          zero_based_counter=False)


scope_mask = np.vstack(scope_pulse_masks).any(axis=0)

masked_raw_stack_times = raw_stack_times[scope_mask]

# %% Construct, if thorsync was stopped early
# n_missing_stack_times = thorimage_settings.n_timepoints - masked_raw_stack_times.size
# ts_stack_offset = (np.arange(n_missing_stack_times) + 1) * np.diff(masked_raw_stack_times).mean()
# stack_times_created = np.append(masked_raw_stack_times,
#                                 ts_stack_offset + masked_raw_stack_times[-1])

n_missing_frame_times = thorimage_settings.n_frames - masked_raw_frame_times.size
ts_frame_offset = (np.arange(n_missing_frame_times) + 1) * np.diff(masked_raw_frame_times).mean()
frame_times_created = np.append(masked_raw_frame_times,
                                ts_frame_offset + masked_raw_frame_times[-1])

stack_times_created = frame_times_created[::thorimage_settings.steps_per_frame]

n_stim = 48

n_missing_olf_ict = n_stim - olf_ict.size
olf_offset = (np.arange(n_missing_olf_ict) + 1) * np.diff(olf_ict).mean()
olf_ict_created = np.append(olf_ict, olf_offset + olf_ict[-1])

n_missing_olf_fct = n_stim - olf_fct.size
olf_offset = (np.arange(n_missing_olf_fct) + 1) * np.diff(olf_fct).mean()
olf_fct_created = np.append(olf_fct, olf_offset + olf_fct[-1])

# %%
stack_times = stack_times_created
frame_times = frame_times_created
olf_ict = olf_ict_created
olf_fct = olf_fct_created

timestamps = dict(
        frame_times=frame_times,
        stack_times=stack_times,
        scope_ict=scope_ict,
        scope_fct=scope_fct,
        olf_ict=olf_ict,
        olf_fct=olf_fct,
        )
# %%
frame_times = masked_raw_frame_times[:thorimage_settings.n_frames]
stack_times = masked_raw_stack_times[:thorimage_settings.n_timepoints]

timestamps = dict(
        frame_times=frame_times,
        stack_times=stack_times,
        scope_ict=scope_ict,
        scope_fct=scope_fct,
        olf_ict=olf_ict,
        olf_fct=olf_fct,
        )
# %%
print('\n--------------------------------------------------')
print(thorimage_xml.relative_to(LEXI_RAW_DIR))
print('--------------------------------------------------')
pp.pprint(thorimage_settings.to_dict(), sort_dicts=False)

print('\nTimestamps info:')
print('-----------------')
for k, v in timestamps.items():
    print(f"{k}: {v.size}")

stack_frame_compat = np.all(stack_times == frame_times[::thorimage_settings.steps_per_frame])
print(f'frame_time compatible with stack_time: {stack_frame_compat}')

# %%
mov_dir = LEXI_PROC_DIR.joinpath(date_imaged,
                                 str(fly_num),
                                 thorimage_name)
print(mov_dir)

# %% Copy Experiment.xml and thorimage_settings.yaml

shutil.copy(thorimage_xml, mov_dir.joinpath("Experiment.xml"))
# %%
with open(mov_dir.joinpath('thorimage_settings.yaml'), 'w') as f:
    yaml.dump(thorimage_settings.to_dict(), f, sort_keys=False)
# %%
np.save(mov_dir.joinpath('timestamps.npy'), timestamps)
# %%
print('\nTimestamps info:')
print('-----------------')
for k, v in timestamps.items():
    print(f"{k}: {v.size}")
# %%
pp.pprint(timestamps, depth=1, compact=True)
