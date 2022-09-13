
def dispatcher(cfg):
    task_name = cfg.task
    model_name = cfg.MODEL.model 

    assert task_name != "none"
    if task_name == "static_link_prediction":
        from .autoencoder_trainer import autoencoder_trainer as autoencoder_trainer_fn
        return autoencoder_trainer_fn
    if task_name == "temporal_link_prediction":
        if model_name in ["EGCNO", "EGCNH"]:
            from .ecgn_trainer import egcn_trainer as egcn_trainer_fn
            return egcn_trainer_fn
        elif model_name == "EULER":
            from .euler_trainer import euler_trainer as euler_trainer_fn
            return euler_trainer_fn
        elif model_name == "DYSAT":
            from .dysat_trainer import dysat_trainer as dysat_trainer_fn
            return dysat_trainer_fn
        elif model_name == "TGAT":
            from .tgat_trainer import tgat_trainer as tgat_trainer_fn
            return tgat_trainer_fn
        elif model_name == "VGRNN":
            from .temporal_graph_trainer import temp_graph_trainer as temp_graph_trainer_fn
            return temp_graph_trainer_fn
            # from .vgrnn_trainer import vgrnn_trainer as vgrnn_trainer_fn
            # return vgrnn_trainer_fn
        elif model_name == "RGCN":
            from .rgcn_trainer import rgcn_trainer as rgcn_trainer_fn
            return rgcn_trainer_fn
        elif model_name == "ProGCN":
            from .progcn_trainer import progcn_trainer as progcn_trainer_fn
            return progcn_trainer_fn
        else:
            from .temporal_graph_trainer import temp_graph_trainer as temp_graph_trainer_fn
            return temp_graph_trainer_fn