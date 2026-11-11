from pathlib import Path
import pandas as pd
import json

manifest_yaml = Path("/local/matrix/Remy-Data/projects/natural_mixtures/"
                     "manifestos/flat_linked_thor_acquisitions.json")

# load manifesto
with open(manifest_yaml) as f:
    all_flat_acqs = json.load(f)

# %%
