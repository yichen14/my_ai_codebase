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

    # Discrete Temporal Network 
    if model_name == "VGRNN":
        from .VGRNN import VGRNN as RGCN_cls
        return RGCN_cls
    if model_name == "EGCNH":
        from .egcn_h import LP_EGCN_h as RGCN_cls
        return RGCN_cls
    if model_name == "EGCNO":
        from .egcn_o import LP_EGCN_o as RGCN_cls
        return RGCN_cls
    if model_name == "DYSAT":
        from .dysat import DySAT as RGCN_cls
        return RGCN_cls
    if model_name == "EULER":
        from .euler import EulerGCN as RGCN_cls
        return RGCN_cls
    
    # Continuous Temporal Network
    if model_name == "TGAT":
        from .tgat import TGAN as RGCN_cls
        return RGCN_cls