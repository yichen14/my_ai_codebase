
def dispatcher(cfg):
    attack_method = cfg.ATTACK.method
    if attack_method == "random":
        from .graph_attack import random_attack_temporal as attack_fn
        return attack_fn
    if attack_method == "meta":
        from .graph_attack import meta_attack_temporal as attack_fn
        return attack_fn
    if attack_method == "dice":
        from .graph_attack import dice_attack_temporal as attack_fn
        return attack_fn
    if attack_method == "node":
        from .graph_attack import node_emb_attack_temporal as attack_fn
        return attack_fn
    if attack_method == "temporal":
        from .graph_attack import temporal_shift_attack as attack_fn
        return attack_fn
    else:
        return None