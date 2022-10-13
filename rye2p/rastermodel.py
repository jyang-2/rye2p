import numpy as np
from pathlib import Path
from scipy.stats import zscore
from rasterviz import norm_traces

import rastermap
from rastermap.mapping import Rastermap

NAS_PROC_DIR = Path("/local/storage/Remy/natural_mixtures/processed_data")

stat_file_list = sorted(list(NAS_PROC_DIR.rglob('combined/stat.npy')))

suite2p_ops = {'n_components': 1, 'n_X': 100, 'alpha': 1., 'K': 1.,
               'nPC': 200, 'constraints': 2, 'annealing': True, 'init': 'pca',
               }0

# %%
stat_file = stat_file_list[0]
iscell, cellprob = np.load(stat_file.with_name('iscell.npy')).T
iscell = iscell == 1

caiman_deconv_results = np.load(stat_file.with_name("caiman_deconv_results.npy"),
                                allow_pickle=True).item()

sp = caiman_deconv_results['C_df'][iscell, :]
spn = norm_traces(sp)
model = Rastermap(**suite2p_ops)
embedding = model.fit_transform(spn)

# def run_rastermap(stat_file):
#     caiman_deconv_results = np.load(stat_file.with_name("caiman_deconv_results.npy"))

#
#     sp = caiman_deconv_results['C_dec']
#     spn = norm_traces(sp)

#
# # fit does not return anything, it adds attributes to model
# # attributes: embedding, u, s, v, isort1
# embedding = model.fit_transform(spn)
# isort1 = np.argsort(embedding[:, 0])
#
# sp = sp[iscell, :]
# spn = norm_traces(sp)
