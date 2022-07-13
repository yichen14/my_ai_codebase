
def dispatcher(cfg):
    attack_method = cfg.ATTACK.method
    if attack_method == "random":
        from .graph_attack import random_attack_temporal as attack_fn
        return attack_fn
    else:
        return None