"""Original model classes, with parameter-free masking of unavailable RF targets."""
import copy
import json
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .vendor.onconet.models.custom_resnet import CustomResnet
from .vendor.onconet.models.hiddens_transfomer import AllImageTransformer
from .vendor.onconet.models.discriminator import Discriminator
from .vendor.onconet.utils.risk_factors import RiskFactorVectorizer

VENDOR = Path(__file__).parent / 'vendor'


def make_args(c, stage, imagenet=False):
    defaults = json.loads((VENDOR / 'argument_defaults.json').read_text())
    keys = json.loads((VENDOR / 'mirai_base.json').read_text())['search_space']['risk_factor_keys'][0].split()
    defaults.update(
        model_name='custom_resnet' if stage == 1 else 'transformer',
        block_layout=[[('BasicBlock', 2)] for _ in range(4)],
        risk_factor_keys=keys, metadata_path=str(VENDOR / 'empty_metadata.json'),
        risk_factor_metadata_path=str(VENDOR / 'empty_risk_factors.json'), dataset='embed',
        cuda=False, model_parallel=False, num_gpus=1, num_shards=1,
        num_chan=3, num_classes=2, use_risk_factors=True, pred_risk_factors=True,
        pretrained_on_imagenet=imagenet, pretrained_imagenet_model_name='resnet18',
        use_precomputed_hiddens=(stage == 2), precomputed_hidden_dim=512,
        transfomer_hidden_dim=c['model']['hidden_dim'], num_heads=c['model']['num_heads'],
        num_layers=1, num_images=4, min_num_images=4, mask_prob=0,
        pred_missing_mammos=False, also_pred_given_mammos=False,
        survival_analysis_setup=True, max_followup=5, pred_both_sides=False,
        pool_name='GlobalMaxPool' if stage == 1 else c['model']['pool'],
        dropout=c[f'stage{stage}']['dropout'], use_pred_risk_factors_at_test=(stage == 2),
        use_pred_risk_factors_if_unk=False, replace_snapshot_pool=False,
        use_spatial_transformer=False, deep_risk_factor_pool=False, make_probs_indep=False,
        use_region_annotation=False, predict_birads=False, adv_on_logits_alone=False,
    )
    return SimpleNamespace(**defaults)


def risk_schema(c):
    a = make_args(c, 1)
    v = RiskFactorVectorizer(a)
    return [{'name': k, 'size': a.risk_factor_key_to_num_class[k],
             'features': v.risk_factor_transformers[k](None, None, just_return_feature_names=True)}
            for k in a.risk_factor_keys]


class RiskTargets(list):
    """Same list-of-tensors forward interface; validity belongs to the loss only."""
    def __init__(self, values, known):
        super().__init__(values)
        self.known = known


def masked_rf_loss(self, hidden, targets):
    img = hidden[:, :-self.length_risk_factor_vector]
    losses = []
    for i, key in enumerate(self.args.risk_factor_keys):
        valid = targets.known[:, i].bool()
        if not valid.any():
            continue
        logits = self._modules[f'{key}_fc'](img)[valid]
        gold = targets[i][valid]
        if logits.shape[-1] == 1:
            losses.append(F.binary_cross_entropy_with_logits(logits, gold))
        else:
            losses.append(F.cross_entropy(logits, gold.argmax(-1)))
    return torch.stack(losses).mean() if losses else img.sum() * 0


def build_model(c, stage, imagenet=False, masked=True):
    a = make_args(c, stage, imagenet)
    model = CustomResnet(a) if stage == 1 else AllImageTransformer(a)
    if masked:
        pool = model._model.pool if stage == 1 else model.pool
        pool.get_pred_rf_loss = types.MethodType(masked_rf_loss, pool)
    return model


def build_adversary(model):
    return Discriminator(copy.deepcopy(model.args))


def architecture(model):
    return {
        'modules': {name: m.__class__.__name__ for name, m in model.named_modules()},
        'state_shapes': {name: list(p.shape) for name, p in model.state_dict().items()},
        'parameter_count': sum(p.numel() for p in model.parameters()),
    }


def risk_loss(logits, batch):
    mask = batch['y_mask'].to(logits.device)
    if mask.sum() <= 0:
        raise ValueError('No supervised horizon in batch')
    return F.binary_cross_entropy_with_logits(logits, batch['y_seq'].to(logits.device),
                                              reduction='none').mul(mask).sum() / mask.sum()


def adversary_inputs(hidden, logits, batch):
    # Match upstream get_adv_loss: one conditioned token per image, detached risk logits.
    b, n, d = hidden.shape
    values = torch.cat((hidden, logits.detach().unsqueeze(1).expand(b, n, 5)), dim=-1).reshape(b*n, d+5)
    known = batch['device_known'].reshape(-1).to(hidden.device).bool()
    labels = batch['device_labels'].reshape(-1).to(device=hidden.device, dtype=torch.long)
    return values, labels, known
