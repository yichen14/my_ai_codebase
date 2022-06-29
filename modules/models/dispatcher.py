def dispatcher(cfg):
    network_name = cfg.MODEL.model
    if network_name == "GAE":
        from models.GAE import GAE as GAE_cls
        return GAE_cls
    if network_name == "GCN":
        from models.GCN import GCN as GCN_cls
        return GCN_cls