from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MappingConfig:
    """Configuration for the online wavemap voxel mapper."""

    voxel_size: float = 0.1       # voxel side length in metres
    min_range: float = 0.1        # minimum depth to integrate (metres)
    max_range: float = 8.0        # maximum depth to integrate (metres)
    occ_threshold: float = 0.6    # log-odds above this => occupied
    free_threshold: float = -1e-5 # log-odds below this => free


@dataclass
class FrontierDetectorConfig:
    """Configuration for occupancy-based 3D frontier detection."""

    height_min: float = 0.1       # Y lower bound — ignore voxels below floor level
    height_max: float = 1.5       # Y upper bound — ignore ceiling frontiers
    min_unknown_neighbors: int = 1 # how many of the 6 face-neighbors must be unknown
    min_free_neighbors: int = 2   # how many face-neighbors must be free (rejects wall voxels)
    subsample_cell: float = 0.5   # coarser grid cell (beta) for subsampling output


@dataclass
class SemanticVoxelMapConfig:
    """Configuration for the sparse semantic voxel map."""

    voxel_size: float = 0.1   # metres per voxel (should match MappingConfig)
    max_range: float = 8.0    # ignore depth beyond this range (metres)
    feature_dim: int = 768    # SigLIP hidden size (set automatically from extractor)
