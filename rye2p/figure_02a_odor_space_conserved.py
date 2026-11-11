from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from rye2p import datadirs
from itertools import combinations
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

prj = 'odor_space_collab'
datadirs.set_prj(prj)

# %%

DATA_DIR = (datadirs.NAS_PRJ_DIR
            .joinpath("analysis_outputs")
            .joinpath('by_imaging_type')
            .joinpath('kc_soma')
            )

concat_rdm_filename = "rdm_concat__kc_soma__megamat__mean_peak__correlation.nc"
# concat_rdm_filename = "rdm_concat__kc_soma__megamat__mean_peak__correlation__trialavg.nc"
ds_rdm_concat = xr.load_dataset(DATA_DIR.joinpath(concat_rdm_filename))

# distance --> similarity
# ---------------------
ds_rdm_concat = 1 - ds_rdm_concat

datefly = [f"{x}/{y}" for x, y in zip(ds_rdm_concat['date_imaged'].to_numpy(),
                                      ds_rdm_concat['fly_num'].to_numpy())]
# %%
ds_rdm_concat = ds_rdm_concat.assign_coords(
        dict(
                abbrev_row=('trial_row', [item.split('@')[0].strip() for item in ds_rdm_concat['stim_col'].to_numpy()]),
                abbrev_col=('trial_col', [item.split('@')[0].strip() for item in ds_rdm_concat['stim_row'].to_numpy()]),
                datefly=('flacq', datefly)
                )
        ).set_index(trial_row='abbrev_row', trial_col='abbrev_col')

ds_rdms_by_fly = ds_rdm_concat.groupby('datefly').mean('flacq')
ds_rdm_mean = ds_rdms_by_fly.mean('datefly')
# %%
# df_plot should have the following columns:
# --------------------------------------------
# Index(['trial_row', 'trial_col', 'datefly', 'stim_row', 'stim_col', 'Fc', 'F',
#        'Fneu', 'spks', 'Fc_smooth', 'Fc_zscore', 'Fc_zscore_smoothed',
#        'Fc_normed', 'Fc_normed_smoothed', 'row_stim', 'row_stim_occ',
#        'col_stim', 'col_stim_occ', 'abbrev_col', 'abbrev_row'],
#       dtype='object')

abbrev_list = list(ds_rdm_concat.trial_row.to_numpy())
pdist_idx = list(combinations(abbrev_list, 2))

df_plot_mean = ds_rdm_mean.to_dataframe()
df_plot_list = [ds_rdms_by_fly.isel(datefly=i).to_dataframe()
                for i in range(ds_rdms_by_fly.dims['datefly'])
                ]
# %%
df_plot_mean = df_plot_mean.loc[pdist_idx, :]
df_plot_list = [df_plot.loc[pdist_idx, :] for df_plot in df_plot_list]

df_plot_mean = df_plot_mean.loc[:, ['Fc_zscore']]
df_plot_list = [df.loc[:, ['Fc_zscore']] for df in df_plot_list]
# %%
df_plot_cat = pd.concat([df_plot_mean] + df_plot_list, axis=1)
df_plot_cat.columns = ['mean'] + [f'fly {i}' for i in range(4)]
df_sim_long = df_plot_cat.sort_values('mean')

cmap = 'RdBu_r'

fig, axarr = plt.subplots(1, 2,
                          figsize=(8, 12),
                          constrained_layout=True)
# LHS: ascending
# ---------------
ax = axarr[0]
sns.heatmap(
        data=df_sim_long,
        vmin=-1,
        vmax=1,
        cmap=cmap,
        square=False,
        linewidths=0,
        xticklabels=True,
        yticklabels=True,
        ax=axarr[0],
        cbar_kws=dict(orientation='horizontal')
        )

ax.set_yticklabels(ax.get_yticklabels(),
                   rotation=0,
                   fontsize=6,
                   )
ax.set_xticklabels(ax.get_xticklabels(),
                   # rotation=45,
                   fontsize=8,
                   )
ax.xaxis.tick_top()

# RHS: unsorted
# ---------------
ax = axarr[1]
sns.heatmap(
        data=df_plot_cat,
        vmin=-1,
        vmax=1,
        cmap=cmap,
        square=False,
        linewidths=0,
        xticklabels=True,
        yticklabels=True,
        ax=ax,
        cbar_kws=dict(orientation='horizontal')
        )
ax.set_yticklabels(ax.get_yticklabels(),
                   rotation=0,
                   fontsize=6,
                   )
ax.set_xticklabels(ax.get_xticklabels(),
                   # rotation=45,
                   fontsize=8,
                   )
ax.xaxis.tick_top()

plt.show()
# %%
if 'trialavg' in concat_rdm_filename:
    pdf_file = datadirs.NAS_PRJ_DIR.joinpath('code_and_data_for_figures',
                                             f'figure_02a_pdistvec__trialavg__{cmap}.pdf')
else:
    pdf_file = datadirs.NAS_PRJ_DIR.joinpath('code_and_data_for_figures',
                                             f'figure_02a_pdistvec__alltrials__{cmap}.pdf')

with PdfPages(pdf_file) as pdf:
    pdf.savefig(fig)
# %%

df_plot = ds_rdms_by_fly.to_dataframe().reset_index()
df_plot = (df_plot
.assign(abbrev_col=[item.split('@')[0].strip() for item in df_plot['stim_col']])
.assign(abbrev_row=[item.split('@')[0].strip() for item in df_plot['stim_row']])
.assign(datefly=pd.Categorical(df_plot['datefly']))
.set_index(['trial_row', 'trial_col'])
.loc[pdist_idx]
)
print(df_plot)
# %% FLY-FLY CORRELATIONS

# make figure imagegrid
# -----------------------------

fig = plt.figure(figsize=(20, 10))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(1, 3),  # creates 2x2 grid of axes
                 axes_pad=0.25,  # pad between axes in inch.
                 cbar_mode='edge',
                 cbar_location='right',
                 cbar_size='5%'
                 )

# plot individual RDMs
# ------------------------
for axrow, cmap, caxs in zip(grid.axes_row,
                             ['vlag', 'RdBu_r'],
                             [grid.cbar_axes[:3], grid.cbar_axes[3:]]
                             ):
    for i, (ax, cax, metric) in enumerate(zip(axrow,
                                              caxs,
                                              ['pearson', 'kendall', 'spearman'])
                                          ):
        sns.heatmap(
                data=df_plot_cat.corr(method=metric),
                vmin=-1,
                vmax=1,
                cmap=cmap,
                cbar_ax=ax.cax,
                square=True,
                linewidths=0,
                xticklabels=True,
                yticklabels=True,
                annot=True,
                ax=ax,
                )
        ax.set(xlabel="", ylabel="")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=12)

        if i == 0:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
        ax.set_title(metric)

plt.show()
#%%
if 'trialavg' in concat_rdm_filename:
    pdf_file = datadirs.NAS_PRJ_DIR.joinpath('code_and_data_for_figures',
                                             f'figure_02a_flycorr__trialavg.pdf')
else:
    pdf_file = datadirs.NAS_PRJ_DIR.joinpath('code_and_data_for_figures',
                                             f'figure_02a_flycorr__alltrials.pdf')

with PdfPages(pdf_file) as pdf:
    pdf.savefig(fig)
# %%
# %%

cmap='vlag'

# make list of dataframes to plot
# ----------------------------------
df_rdm_list = [ds_rdms_by_fly['Fc_zscore'].isel(datefly=i).to_pandas()
               for i in range(ds_rdms_by_fly.dims['datefly'])]
df_rdm_list = [df for df in df_rdm_list]

# mean dataframe
# ---------------------------------
df_rdm_mean = ds_rdms_by_fly.mean('datefly')['Fc_zscore'].to_pandas()

# add mean dataframe to list
# --------------------------------
df_rdm_list.append(df_rdm_mean)

# make figure imagegrid
# -----------------------------
fig = plt.figure(figsize=(20, 8))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(1, 5),  # creates 2x2 grid of axes
                 axes_pad=0.25,  # pad between axes in inch.
                 cbar_mode='edge',
                 cbar_size='5%'
                 )

# plot individual RDMs
# ------------------------
for i, (ax, cax, df) in enumerate(zip(grid.axes_all,
                                      grid.cbar_axes,
                                      df_rdm_list)
                                  ):
    sns.heatmap(
            data=df,
            vmin=-1,
            vmax=1,
            cmap=cmap,
            cbar_ax=cax,
            square=True,
            linewidths=0,
            xticklabels=3,
            yticklabels=3,
            ax=ax,
            )
    ax.set(xlabel="", ylabel="")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    if i == 0:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    ax.set_title(str(i))

grid.axes_all[-1].set_title('mean')
plt.show()
# %%
if 'trialavg' in concat_rdm_filename:
    pdf_file = datadirs.NAS_PRJ_DIR.joinpath('code_and_data_for_figures',
                                             f'figure_02a_all_rdms__trialavg__{cmap}.pdf')
else:
    pdf_file = datadirs.NAS_PRJ_DIR.joinpath('code_and_data_for_figures',
                                             f'figure_02a_all_rdms__alltrials__{cmap}.pdf')

with PdfPages(pdf_file) as pdf:
    pdf.savefig(fig)
# %%
