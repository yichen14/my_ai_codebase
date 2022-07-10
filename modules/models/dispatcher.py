def dispatcher(cfg):
    network_name = cfg.MODEL.model
    encoder = cfg.MODEL.encoder
    if network_name == "GAE":
        from .autoencoder import GAE as GAE_cls
        if encoder == "GCN":
            from .encoders import GCNEncoder as encoder_cls
        if encoder == "linear":
            from .encoders import LinearEncoder as encoder_cls
        return GAE_cls, encoder_cls
    if network_name == "VGAE":
        from .autoencoder import VGAE as VGAE_cls
        if encoder == "GCN":
            from .encoders import VariationalGCNEncoder as encoder_cls
        if encoder == "linear":
            from .encoders import VariationalLinearEncoder as encoder_cls
        return VGAE_cls, encoder_cls
    if network_name == "GCN":
        from models.GCN import GCN as GCN_cls
        return GCN_cls

    # Temporal Network 
    if network_name == "EGCNH":
        from .RecurrentGCN import RecurrentGCN_EGCNH as RGCN_cls
        return RGCN_cls
    if network_name == "EGCNO":
        from .RecurrentGCN import RecurrentGCN_EGCNO as RGCN_cls
        return RGCN_cls
    if network_name == "DCRNN":
        from .RecurrentGCN import RecurrentGCN_DCRNN as RGCN_cls
        return RGCN_cls
    if network_name == "GCLSTM":
        from .RecurrentGCN import RecurrentGCN_GCLSTM as RGCN_cls
        return RGCN_cls