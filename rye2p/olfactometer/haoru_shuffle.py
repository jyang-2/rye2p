from pathlib import Path
import yaml
import random
import copy
from datetime import datetime

OLF_CONFIG_DIR = Path("/local/tensor/Haoru/rotation_project/olfactometer_configs")

default_pin_2_balance = {
    37: 42,
    38: 42,
    39: 42,
    40: 42,
    41: 42,
    45: 50,
    46: 50,
    47: 50,
    48: 50,
    49: 50}


def yaml_filename_now():
    """Generate yaml filename with current date and time"""
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{now_str}_stimuli_.yaml"


def get_paraffin_pin(pins2odors):
    for pin, odor in pins2odors.items():
        if odor['name'] in ['pfo', 'paraffin']:
            return pin
    raise ValueError("No paraffin pin")


def make_shuffled_single_pin_sequence(pins2odors, n_repeats=3, paraffin_first=True):
    """Make randomized pin sequence from pin map of odors.

    All odors must be presented once, before being repeated.

    Args:
        pins2odors (dict): pin map of odors.
        n_repeats (int): number of stimulus repeats
        paraffin_first (bool): if true, always start a shuffle block with paraffin
        pin2balance (dict): map of odor pins to the balance pin

    Returns:
        pin_sequence

    """
    if paraffin_first:
        pfo_pin = get_paraffin_pin(pins2odors)
        pins_to_shuffle = [item for item in pins2odors.keys() if item != pfo_pin]
    else:
        pins_to_shuffle = list(pins2odors.keys())

    shuffled_blocks = [random.sample(pins_to_shuffle, len(pins_to_shuffle))
                       for _ in range(n_repeats)]

    if paraffin_first:
        shuffled_blocks = [[pfo_pin] + block for block in shuffled_blocks]

    return [item for block in shuffled_blocks for item in block]


def balance_pins(single_pin_sequence, pin2balance):
    return [[pin, pin2balance[pin]] for pin in single_pin_sequence]


def make_pin_groups(balanced_pin_sequence):
    return [dict(pins=pins) for pins in balanced_pin_sequence]


if __name__ == "__main__":
    ref_yaml_name = '20250115_164737_stimuli.yaml'
    ref_yaml_file = OLF_CONFIG_DIR.joinpath(ref_yaml_name)

    # Load reference .yaml file
    with open(ref_yaml_file, 'r') as f:
        olf_config = yaml.safe_load(f)

    # Make new stimulus sequence
    shuffled_single_pins = make_shuffled_single_pin_sequence(olf_config['pins2odors'],
                                                             n_repeats=3,
                                                             paraffin_first=True)
    shuffled_balanced_pins = balance_pins(shuffled_single_pins, default_pin_2_balance)
    pin_groups = make_pin_groups(shuffled_balanced_pins)

    # Make new config with shuffled non-consecutive repeats
    shuffled_olf_config = copy.deepcopy(olf_config)
    shuffled_olf_config['pin_sequence']['pin_groups'] = pin_groups

    write_new_yaml = True
    if write_new_yaml:
        # save new config file with shuffled stimuli
        new_yaml_file = OLF_CONFIG_DIR.joinpath(yaml_filename_now())

        with open(new_yaml_file, 'w') as f:
            yaml.dump(shuffled_olf_config, f, sort_keys=False)

        print(f"New olf_config created:")
        print('========================')
        print(f"reference yaml:\n\t{ref_yaml_file}")
        print(f"\nshuffled yaml:\n\t{new_yaml_file}")
