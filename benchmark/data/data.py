"""
    File to load dataset based on user control from main file
"""
from data.superpixels import SuperPixDataset
from data.molecules import MoleculeDataset
from data.TUs import TUsDataset
from data.SBMs import SBMsDataset
from data.TSP import TSPDataset
from data.CitationGraphs import CitationGraphsDataset
from data.TOX import TOXDataset

def LoadData(DATASET_NAME):
    """
        This function is called in the main.py file
        returns:
        ; dataset object
    """
    toxic_reps = ['PBT_Repn1', 'PBT_Repn2', 'PBT_Repn3', 'PBT_Repn4', 'PBT_Rep1', 'PBT_Rep2', 'PBT_Rep3', 'PBT_Rep4', 'CMR_Rep1', 'CMR_Rep2', 'CMR_Rep3', 'CMR_Rep4']

    # handling for (TOX) molecule dataset
    if DATASET_NAME in toxic_reps:
        return TOXDataset(DATASET_NAME)

    # handling for MNIST or CIFAR Superpixels
    if DATASET_NAME == 'MNIST' or DATASET_NAME == 'CIFAR10':
        return SuperPixDataset(DATASET_NAME)

    # handling for (ZINC) molecule dataset
    if DATASET_NAME == 'ZINC':
        return MoleculeDataset(DATASET_NAME)

    # handling for the TU Datasets
    TU_DATASETS = ['COLLAB', 'ENZYMES', 'DD', 'PROTEINS_full']
    if DATASET_NAME in TU_DATASETS:
        return TUsDataset(DATASET_NAME)

    # handling for SBM datasets
    SBM_DATASETS = ['SBM_CLUSTER', 'SBM_PATTERN']
    if DATASET_NAME in SBM_DATASETS:
        return SBMsDataset(DATASET_NAME)

    # handling for TSP dataset
    if DATASET_NAME == 'TSP':
        return TSPDataset(DATASET_NAME)

    # handling for the CITATIONGRAPHS Datasets
    CITATIONGRAPHS_DATASETS = ['CORA', 'CITESEER', 'PUBMED']
    if DATASET_NAME in CITATIONGRAPHS_DATASETS:
        return CitationGraphsDataset(DATASET_NAME)
