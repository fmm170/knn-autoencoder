from typing import List
from omegaconf import MISSING
from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass, ChoiceEnum


KEYS_DTYPE_CHOICES = ChoiceEnum(["fp16", "fp32"])


@dataclass
class BaseKnnConfig(FairseqDataclass):
    datastore: str = field(
        default=MISSING,
        metadata={
            "help": "path to datastore directory"
        }
    )
    datastore_size: int = field(
        default=0,
        metadata={
            "help": "the number of tokens in datastore"
        }
    )
    keys_dimension: int = field(
        default=1024,
        metadata={
            "help": "the feature dimension of datastore keys"
        }
    )
    keys_dtype: KEYS_DTYPE_CHOICES = field(
        default="fp16",
        metadata={
            "help": "keys dtype of the datastore"
        }
    )
    load_keys: bool = field(
        default=False,
        metadata={
            "help": "whether to load datastore keys"
        }
    )
    load_value_weights: bool = field(
        default=False,
        metadata={
            "help": "whether to load the weights of datastore values"
        }
    )
    nprobe: int = field(
        default=32,
        metadata={
            "help": "the number of clusters to query"
        }
    )
    knn_device_id: List[int] = field(
        default_factory=lambda: [0],
        metadata={
            "help": "ID of GPU device used for (approximate) knn search. "
            "a single negtive number means using CPU instead of GPU. "
            "note that this device can be different from the one used for translation."
            "if there is more than one number in `knn_device_id`, all numbers must "
            "be greater or equal to zero and the faiss index will be sharded across "
            "the GPU devices specified by `knn_device_id`. "
            "eg., --knn-device-id '1,2,3' "
        }
    )
    knn_fp16: bool = field(
        default=False,
        metadata={
            "help": "whether to perform intermediate calculations in float16 during (approximate) knn search"
        }
    )
    move_to_memory: bool = field(
        default=False,
        metadata={
            "help": "whether to move the datastore into CPU memory"
        }
    )
    module_to_capture: str = field(
        default="layers[-1]",
        metadata={
            "help": "the outputs of the which module in decoder to be captured. "
            "the default module is the last layer of decoder"
        }
    )
    saving_mode: bool = field(
        default=False,
        metadata={
            "help": "whether to use saving mode. "
            "the knn search setup process will be skipped in saving mode. "
            "saving mode is usually used when saving datastore"
        }
    )
    domain : str = field(
        default="medical",
        metadata={
            "help": "multidomain"
        }
    )
    total_tokens : int = field(
        default=0,
        metadata={
            "help" : "training set total tokens size"
        }
    )

@dataclass
class KnnConfig(BaseKnnConfig):
    num_neighbors: int = field(
        default=1,
        metadata={
            "help": "the number of neighbors to retrieve"
        }
    )
    lambda_value: float = field(
        default=0.5,
        metadata={
            "help": "hyperparameter used for interpolation of kNN and MT probability distributions"
        }
    )
    temperature_value: float = field(
        default=10,
        metadata={
            "help": "hyperparameter used for flattening the kNN probability distribution"
        }
    )
    pos_sample : str = field(
        default= "/users10/lhuang/xiaokenaifan/knn-models/examples/knnmt",
        metadata={
            "help":"search k positive sample"
        }
    )
    neg_sample : str = field(
        default= "/users10/lhuang/xiaokenaifan/knn-models/examples/knnmt",
        metadata={
            "help":"search k negative sample"
        }
    )
    input_dim : int = field(
        default= 1024,
        metadata={
            "help": "model output dimension"
        }
    )
    output_dim : int = field(
        default=512,
        metadata={
            "help":"low demension"
        }
    )
    auto_encoder : str = field(
        default= "/users10/lhuang/xiaokenaifan/ConAE/train/MSMARCO/multi-domain-data-datastore/medical/save_model_512/model.best.pt",
        metadata={
            "help":"auto_encoder checkpoint"
        }
    )
    whether_generate_datastore : bool = field(
        default= False,
        metadata={
            "help":"whether to generate datastore"
        }
    )


@dataclass
class AdaptiveKnnConfig(BaseKnnConfig):
    num_neighbors: int = field(
        default=1,
        metadata={
            "help": "the number of neighbors to retrieve"
        }
    )
    temperature_value: float = field(
        default=10,
        metadata={
            "help": "hyperparameter used for flattening the kNN probability distribution"
        }
    )
    ae_network_input_dim: int = field(
        default=1024,
        metadata={
            "help":"ae model input dimension"
        }
    )
    ae_network_output_dim: int = field(
        default=64,
        metadata={
            "help": "ae model output dimension"
        }
    )
    meta_k_hidden_size: int = field(
        default=32,
        metadata={
            "help": "hidden size of meta-k network"
        }
    )
    meta_k_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout of meta-k network"
        }
    )
