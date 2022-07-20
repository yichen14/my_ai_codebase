def dispatcher(cfg):
    model_name = cfg.MODEL.model
    encoder = cfg.MODEL.encoder
    if model_name == "GAE":
        from .autoencoder import GAE as GAE_cls
        if encoder == "GCN":
            from .encoders import GCNEncoder as encoder_cls
        if encoder == "linear":
            from .encoders import LinearEncoder as encoder_cls
        return GAE_cls, encoder_cls
    if model_name == "VGAE":
        from .autoencoder import VGAE as VGAE_cls
        if encoder == "GCN":
            from .encoders import VariationalGCNEncoder as encoder_cls
        if encoder == "linear":
            from .encoders import VariationalLinearEncoder as encoder_cls
        return VGAE_cls, encoder_cls
    if model_name == "GCN":
        from models.GCN import GCN as GCN_cls
        return GCN_cls

    # Temporal Network 
    if model_name == "EGCNH":
        from .RecurrentGCN import RecurrentGCN_EGCNH as RGCN_cls
        return RGCN_cls
    if model_name == "EGCNO":
        from .egcn_o import LP_EGCN_o as RGCN_cls
        return RGCN_cls
    if model_name == "DCRNN":
        from .RecurrentGCN import RecurrentGCN_DCRNN as RGCN_cls
        return RGCN_cls
    if model_name == "GCLSTM":
        from .RecurrentGCN import RecurrentGCN_GCLSTM as RGCN_cls
        return RGCN_cls
    if model_name == "VGRNN":
        from .VGRNN import VGRNN as RGCN_cls
        return RGCN_cls