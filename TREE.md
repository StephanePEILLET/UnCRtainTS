# Project File Tree

## Root
- `temp_test.py`: Temporary testing script?
- `standalone_dataloader.py`: Standalone script for data loading?
- `test_dataloader.py`: Script to test the dataloader.
- `standalone_circa_dataloader.py`: Standalone script for CIRCA data loading?

## Data (`data/`)
- `uncrtaints_adapter.py`: Adapter for UnCRtainTS to CIRCA.
- `data_module.py`: PyTorch Lightning DataModule or similar structure.
- `dataLoader.py`: Main data loading logic (SEN12MSCR, SEN12MSCRTS).
- `circa_dataloader.py`: Dataloader specific to CIRCA dataset.
- `__init__.py`: Package initialization.

### Constants (`data/constants/`)
- `circa_constants.py`: Constants for CIRCA dataset.
- `__init__.py`: Package initialization.

### Utils (`data/utils/`)
- `sampling_functions.py`: Functions for sampling data.
- `process_functions.py`: Data processing functions.
- `__init__.py`: Package initialization.

## Model (`model/`)
- `train_reconstruct.py`: Main training script for reconstruction.
- `test_reconstruct.py`: Testing script for reconstruction.
- `test_reconstruct_circa.py`: Testing script for reconstruction on CIRCA.
- `ensemble_reconstruct.py`: Script for ensemble reconstruction.
- `iterate_proposal.py`: **NEW** Modular iteration logic (IterationManager).
- `imputation.py`: Imputation logic.
- `parse_args.py`: Argument parsing utilities.

### Source (`model/src/`)
- `utils.py`: **UPDATED** Shared utility functions (prepare_data, logging, plotting).
- `losses.py`: Loss functions.
- `model_utils.py`: Model loading/saving utilities.
- `learning/`: Learning related modules (metrics, weight_init).
- `backbones/`: Neural network backbone definitions (UNet, ConvLSTM, etc.).

## Utilities (`util/`)
- `utils.py`: General utilities.
- `detect_cloudshadow.py`: Cloud shadow detection.
- `pre_compute_data_samples.py`: Pre-computation script.
- `hdf5converter/`: Tools to convert data to HDF5.
- `pytorch_ssim/`: SSIM implementation.

## Legacy / Old (`dataloader_CIRCA_old/`)
- Contains previous dataloader implementations and tools.
