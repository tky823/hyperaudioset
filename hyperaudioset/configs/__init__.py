from dataclasses import dataclass


@dataclass
class DistributedConfig:
    enable: bool
    nodes: int
    nproc_per_node: int
    backend: str
    init_method: str

    # multi-node training parameters
    rdzv_id: int
    rdzv_backend: str
    rdzv_endpoint: str
    max_restarts: int


@dataclass
class CuDNNConfig:
    benchmark: bool
    deterministic: bool


@dataclass
class AMPConfig:
    enable: bool
    dtype: str


@dataclass
class IndexerConfig:
    pass


@dataclass
class TrainingDatasetConfig:
    pass


@dataclass
class EvaluationDatasetConfig:
    pass


@dataclass
class TrainingDataLoaderConfig:
    pass


@dataclass
class EvaluationDataLoaderConfig:
    pass


@dataclass
class DatasetConfig:
    train: TrainingDatasetConfig
    evaluate: EvaluationDatasetConfig


@dataclass
class DataLoaderConfig:
    train: TrainingDataLoaderConfig
    evaluate: EvaluationDataLoaderConfig


@dataclass
class _OptimizerConfig:
    pass


@dataclass
class _LRSchedulerConfig:
    pass


@dataclass
class SystemConfig:
    seed: int
    distributed: DistributedConfig
    cudnn: CuDNNConfig
    amp: AMPConfig
    accelerator: str


@dataclass
class DataConfig:
    indexer: IndexerConfig
    dataset: DatasetConfig
    dataloader: DataLoaderConfig
    num_embedddings: int


@dataclass
class ModelConfig:
    pass


@dataclass
class CriterionConfig:
    pass


@dataclass
class OptimizerConfig:
    optimizer: _OptimizerConfig
    lr_scheduler: _LRSchedulerConfig


@dataclass
class Config:
    system: SystemConfig
    data: DataConfig
    model: ModelConfig
    criterion: CriterionConfig
    optimizer: OptimizerConfig
    exp_dir: str
    epochs: int
    tensorboard_dir: str
    root: str
    depth: int
