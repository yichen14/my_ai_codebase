
def dispatcher(cfg):
    attack_method = cfg.ATTACK.method
    if attack_method == "random":
        from .graph_attack import generate_random_attack as attack_fn
        return attack_fn
    else:
        return None