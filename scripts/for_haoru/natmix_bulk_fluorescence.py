import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils2p
from pathlib import Path

# 5606 timestamps
proc_dir = Path("/local/matrix/Remy-Data/projects/natural_mixtures/processed_data/"
                "2022-07-12/2/kiwi_components_again_with_partial_and_probes")

fluors = []
for i in range(3):
    df = pd.read_csv(proc_dir / f"bulk_fluorescence/Data_ROI_{i:03d}.csv",
                     index_col=0)
    if i > 0:
        df['frame'] = df['frame'] + fluors[i - 1]['frame'].max() + 1

    fluors.append(df)

df_bulk_fluor = pd.concat(fluors, axis=0, ignore_index=True)
# %% Load label image
label_img = utils2p.load_img(proc_dir / "bulk_fluorescence/Labels.tif")

# %% Load tiff files
soma = []
calyx = []

tiff_files = sorted(list((proc_dir / "bulk_fluorescence/stk").glob("*.tif")))

for file in tiff_files:
    print(file.name)
    stack = utils2p.load_img(file)
    n_frames = stack.shape[0]

    stack_flat = stack.reshape(stack.shape[0], -1)

    # calyx label = 1
    # soma label = 2
    calyx_f = stack_flat[:, label_img.flatten() == 1].mean(axis=1)
    soma_f = stack_flat[:, label_img.flatten() == 2].mean(axis=1)

    soma.append(soma_f)
    calyx.append(calyx_f)
# %%
soma_timeseries = np.concatenate(soma)
calyx_timeseries = np.concatenate(calyx)
# %%
df_bulk_fluorescence = pd.DataFrame.from_dict(
        {'soma': soma_timeseries, 'calyx': calyx_timeseries},
        orient='columns'
        )
# %%
df_bulk_fluorescence.to_csv(
        proc_dir / "bulk_fluorescence/by_remy/df_bulk_fluorescence.csv",
        index=False)
# %%

timestamps = np.load(proc_dir / "timestamps.npy",
                     allow_pickle=True).item()
# %%
fig, axarr = plt.subplots(2, 1, figsize=(12, 4))

for ax, region in zip(axarr.flat, ['calyx', 'soma']):
    ax.plot(timestamps['stack_times'],
            df_bulk_fluorescence[region],
            lw=0.5);

    for ict in timestamps['olf_ict']:
        ax.axvline(ict, zorder=0, alpha=0.5, lw=0.5)
    ax.set_title(region)
plt.show()
