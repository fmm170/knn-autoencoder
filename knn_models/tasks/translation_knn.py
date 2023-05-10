import logging
import torch
from functools import partial
from dataclasses import dataclass
from fairseq.tasks.translation import (
    TranslationTask,
    TranslationConfig,
)
from fairseq.tasks import register_task
from fairseq.dataclass import FairseqDataclass
from knn_models.dataclass import (
    KnnConfig,
    DimReduceConfig,
)
from knn_models.hook_utils import (
    ForwardHook,
    DimReduceForwardHook,
)
from knn_models.knn_utils import (
    KnnSearch,
    get_captured_module,
    get_normalized_probs,
)
from ..my_distill_model import my_ConAE_model
logger = logging.getLogger(__name__)
@dataclass
class TranslationKnnConfig(TranslationConfig):
    """config for nearest neighbor machine translation"""
    knn_config: KnnConfig = KnnConfig()
    dim_reduce_config: DimReduceConfig = DimReduceConfig()


@register_task("translation_knn", dataclass=TranslationKnnConfig)
class TranslationKnnTask(TranslationTask):
    """task for nearest neighbor machine translation"""
    def __init__(self, cfg: TranslationKnnConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.knn_search = KnnSearch(cfg.knn_config)
        if cfg.dim_reduce_config.dim_reduce_method is None:
            self.forward_hook = ForwardHook(cfg.knn_config)
        else:
            self.forward_hook = DimReduceForwardHook(cfg.dim_reduce_config)

    def build_model(self, cfg: FairseqDataclass, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)

        assert hasattr(model, "decoder"), \
            "TranslationKnnTask only supports the model with decoder! " \
            f"There is no decoder in {model.__class__.__name__}."
        
        # collect outputs from the specified module in decoder as the datastore keys
        captured_module_name = self.cfg.knn_config.module_to_capture
        captured_module = get_captured_module(model.decoder, captured_module_name)
        # captured_module.register_forward_hook(self.forward_hook.forward_hook_function)
        ######注释这里######
        logger.info("加载auto-encoder")
        AE_model = my_ConAE_model(self.cfg.knn_config).cuda()
        AE_model.load_state_dict(torch.load(self.cfg.knn_config.auto_encoder)['model'], strict=False)
        # logger.info(AE_model)
        AE_model.eval()
        import functools
        captured_module.register_forward_hook(functools.partial(self.forward_hook.forward_hook_function,AE_model))
        ######注释这里######

        # rewrite `get_normalized_probs` function to support kNN augmented NMT
        model.get_normalized_probs = partial(get_normalized_probs, self, model)
        return model
