import logging
import torch
from functools import partial
from dataclasses import dataclass

from fairseq import search
from fairseq.tasks.translation import (
    TranslationTask,
    TranslationConfig,
)
from fairseq.tasks import register_task
from fairseq.dataclass import FairseqDataclass
from knn_models.dataclass import (
    KnnConfig,
    DimReduceConfig,
    LlmaConfig,
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
    # LLMA魔改
    llma_config: LlmaConfig = LlmaConfig()


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

        # LLMA魔改
        # 在这个地方得把cfg中需要的信息在这里留下来，否则就要动generate.py
        # 因为generate在构建生成器的时候只会传入cfg.generation

        self.llma: bool = getattr(cfg, "llma", False)
        self.llma_criterion: bool = getattr(cfg, "llma_criterion", "prefix")
        self.llma_prefix_length: int = getattr(cfg, "llma_prefix_length", 5)
        self.llma_suffix_length: int = getattr(cfg, "llma_suffix_length", 5)
        self.llma_threshold: float = getattr(cfg, "llma_threshold", 0.8)

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
        captured_module.register_forward_hook(functools.partial(self.forward_hook.forward_hook_function, AE_model))
        ######注释这里######

        # rewrite `get_normalized_probs` function to support kNN augmented NMT
        model.get_normalized_probs = partial(get_normalized_probs, self, model)
        return model

    def build_generator(
            self,
            models,
            args,
            seq_gen_cls=None,
            extra_gen_cls_kwargs=None,
            prefix_allowed_tokens_fn=None,
    ):

        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)


        from knn_models.llma_utils import SequenceGeneratorWithLLMA
        return SequenceGeneratorWithLLMA(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
            llma=self.llma,
            llma_criterion=self.llma_criterion,
            llma_prefix_length=self.llma_prefix_length,
            llma_suffix_length=self.llma_suffix_length,
            llma_threshold=self.llma_threshold

        )
