from pathlib import Path
from attrs import define, field
from typing import List, Optional, Tuple, Union
import yaml


@define
class OdorVial:
    name: str
    log10_conc: float
    abbrev: str

    def to_stim_str(self) -> str:
        return f"{self.abbrev} @ {self.log10_conc}"

@define
class Generator:
    pre_pulse_s: int
    pulse_s: int
    post_pulse_s: int
    odors: List[OdorVial]


# %%
yaml_file = Path("/local/tensor/Remy/natural_mixtures/olfactometer_configs/kiwi.yaml")

with open(yaml_file, 'r') as f:
    generator_data = yaml.load(f, Loader=yaml.FullLoader)

generator_data
# %%
import cattrs
cattrs.structure(generator_data, Generator)
