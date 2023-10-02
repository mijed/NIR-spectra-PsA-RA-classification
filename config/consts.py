# TODO: refactor paths with pathlib
# Data Management
DATA_PATH = '../../data_src/'
LABELS_FILENAME = 'labelki.txt'
NIR_SPECTRA_FILENAME = 'nir.xls'

LABELS_MAPPER = {
    'RZS': 0,
    'ŁZS': 1,
    'K': 2
}

# Model config
LR = 1e-4

# How many points to discard from NIR spectra
NIR_SPECTRA_CUT_OFF = 30
