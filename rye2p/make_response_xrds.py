from pathlib import Path
import numpy as np
import xarray as xr
import pydantic
import typing
from typing import List
import yaml
#%%
class OdorSpaceFly(pydantic.BaseModel):
    fly_id: int
    date_imaged: str
    fly_num: int
    genotype: str
    date_eclosed: str
    sex: str
    expt_list: typing.List[int]

#%%
DBX_PROC_DIR = Path("/media/remy/remy-storage/Remy's Dropbox Folder/"
                    "HongLab @ Caltech Dropbox/Remy/mb_odor_space/processed-data")

NAS_PROC_DIR = Path("/local/storage/Remy/mb_odor_space/xarray_data")



# %% load mb_odor_space fly manifest
# ==================================

MANIFEST_DIR = DBX_PROC_DIR.with_name('manifestos')


with open(MANIFEST_DIR.joinpath('fly_manifest.yaml'), 'r') as f:
    fly_manifest = yaml.safe_load(f)

fly_manifest = pydantic.parse_obj_as(List[OdorSpaceFly], fly_manifest)
print(fly_manifest)
#%%

#NAS_PROC_DIR = DBX_PROC_DIR.with_name('xarray_data')

# get list of .npy files
file_list = sorted(list(DBX_PROC_DIR.rglob('response_xrds.npy')))

for file in file_list:
    response_xrds_npy = np.load(file, allow_pickle=True).item()
    ds = xr.Dataset.from_dict(response_xrds_npy)
    ds.coords['trial'] = np.arange(ds.dims['trial'])

    # fix attributes
    attrs = dict(fly_id=ds.attrs['fly_id'],
                 expt_id=ds.attrs['mov_id'],
                 std_thr=ds.attrs['response_options']['std_thr'],
                 thr=ds.attrs['response_options']['thr'],
                 max_k=ds.attrs['response_options']['max_k'])

    ds.attrs = attrs

    save_file = file.relative_to(DBX_PROC_DIR).with_name('xrds_respvecs.nc')
    save_file = NAS_PROC_DIR / save_file

    save_dir = save_file.parent
    if not save_dir.is_dir():
        save_dir.mkdir(parents=True)

    print(save_file)

    ds.to_netcdf(save_file)



