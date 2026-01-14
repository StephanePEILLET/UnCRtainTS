"""
    Permet de passer un dataset CircaPatchDataSet au format HDF5.
"""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))
import h5py
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import numpy as np
from tqdm.auto import tqdm, trange

from dataloader_CIRCA_old.datasets import CircaPatchDataSet


def pytorch_dict_to_hdf5(dataset, output_file):
    """
    Convertit un Dataset PyTorch retournant des dictionnaires en fichier HDF5
    
    Args:
        dataset: Instance de torch.utils.data.Dataset
        output_file: Chemin vers le fichier HDF5 de sortie
    """

    with h5py.File(output_file, 'w') as hf:
        # Créer des groupes pour organiser les données
        data_group = hf.create_group('data')
        meta_group = hf.create_group('metadata')
        
        # Parcourir tous les échantillons du dataset
        for i in trange(len(dataset)):
            sample = dataset[i]
            # Créer un groupe pour cet échantillon
            sample_group = data_group.create_group(f'sample_{i}')
            # Stocker chaque tenseur du dictionnaire
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    # Convertir le tenseur PyTorch en numpy array et le stocker
                    sample_group.create_dataset(key, data=value.numpy())
                elif isinstance(value, dict):
                    # Gérer les métadonnées imbriquées
                    meta_subgroup = meta_group.create_group(f'sample_{i}_{key}')
                    for meta_key, meta_value in value.items():
                        if isinstance(meta_value, (str, int, float)):
                            meta_subgroup.attrs[meta_key] = meta_value
                
        # Ajouter des attributs globaux au fichier
        hf.attrs['num_samples'] = len(dataset)
        hf.attrs['dataset_type'] = 'converted_from_pytorch'


def pytorch_dict_2_hdf5(dataset, output_file, num_workers=8):
    dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            prefetch_factor=2,
    )
    progress_bar = tqdm(total=len(dataset))
    with h5py.File(output_file, 'w') as hf:
        data_group = hf.create_group('data')
        meta_group = hf.create_group('metadata')
        hf.attrs['num_samples'] = len(dataset)

        for i, sample in enumerate(dataloader):
            sample_group = data_group.create_group(f'sample_{i}')
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    sample_group.create_dataset(key, data=value.numpy())
                elif isinstance(value, dict):
                    meta_subgroup = meta_group.create_group(f'sample_{i}_{key}')
                    for meta_key, meta_value in value.items():
                        if isinstance(meta_value, (str, int, float)):
                            meta_subgroup.attrs[meta_key] = meta_value               
            progress_bar.update(1)


if __name__ == "__main__":
    store_dai = Path("/home/SPeillet/Partage/store-dai")
    path_dataset_circa = store_dai / "projets/pac/3str/EXP_2"
    data_optique = path_dataset_circa / "Data_Raster" / "optique_dataset"
    data_radar = path_dataset_circa / "Data_Raster" / "radar_dataset_v4"
    PATCH_SIZE = 256
    OVERLAP = 0
    SUBSET = True
    SUBSIZE = 100

    dataset = CircaPatchDataSet(
        data_optique=data_optique,
        data_radar=data_radar,
        patch_size=PATCH_SIZE,
        overlap=OVERLAP,
        load_dataset="datasetCIRCAUnCRtainTS.csv"
    )

    output_file =  store_dai / "tmp/speillet/patches_circa.hdf5"

    if SUBSET:
        dataset = Subset(dataset, np.arange(SUBSIZE))

    # Convertir le dataset PyTorch en HDF5
    if not output_file.exists():
        print("Conversion du dataset PyTorch en HDF5...")
        pytorch_dict_2_hdf5(dataset, output_file, num_workers=8) # Celle là ne marche pas 
        # pytorch_dict_to_hdf5(dataset, output_file) # cette fonction marche 
        print(f"Dataset converti et enregistré dans {output_file}")


