import numpy as np
import pandas as pd
from config.consts import DATA_PATH, NIR_SPECTRA_FILENAME, LABELS_FILENAME, NIR_SPECTRA_CUT_OFF, LABELS_MAPPER
import os


class DataManager:
    data_path: str = os.path.join(DATA_PATH, NIR_SPECTRA_FILENAME)
    labels_path: str = os.path.join(DATA_PATH, LABELS_FILENAME)

    def load_data(self) -> tuple[np.ndarray]:
        # Read NIR spectra data_src
        nir_df = pd.read_excel(self.data_path)
        nir_df = nir_df.iloc[NIR_SPECTRA_CUT_OFF:-NIR_SPECTRA_CUT_OFF, :]
        # Transpose to samples x spectra
        nir_df = nir_df.T

        labels_df = pd.read_csv(self.labels_path, sep=',', usecols=['label'])
        labels_df = labels_df['label'].map(LABELS_MAPPER)

        return nir_df.to_numpy(), labels_df.to_numpy()

