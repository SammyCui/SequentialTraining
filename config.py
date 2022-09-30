from typing import List, Optional, Tuple
from pydantic import BaseModel


def none_or_str(value):
    if value == 'None':
        return None
    return value


class Config(BaseModel):
    regimens: List[str]
    # dataset
    dataset_name: str
    train_image_path: List[str]
    train_annotation_path: Optional[List[str]] = None
    test_annotation_path: Optional[str] = None
    test_image_path: str
    train_size: Optional[int] = None
    test_size: Optional[int] = None
    num_classes: Optional[int] = None
    cls_to_use: Optional[List[str]] = None

    # image
    input_size: Tuple[int, int]
    sizes: List[float]
    resize_method: str
    background: Optional[str] = 'black'

    # training
    model: str
    epoch: int
    min_lr: float
    n_folds: int
    n_folds_to_use: int
    early_stop_patience: int = 20
    max_norm: Optional[int] = None
    reset_lr: Optional[bool] = False
    optim_kwargs: dict
    scheduler_kwargs: dict
    device: str
    batch_size: Optional[int] = 128
    num_workers: Optional[int] = 16

    result_path: str
    save_progress_ckpt: bool
    save_result_ckpt: bool

    random_seed: Optional[int] = 40
