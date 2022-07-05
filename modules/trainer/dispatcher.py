
def dispatcher(cfg):
    task_name = cfg.task
    assert task_name != "none"
    if task_name == "static_link_prediction":
        from .autoencoder_trainer import autoencoder_trainer as autoencoder_trainer_fn
        return autoencoder_trainer_fn
    if task_name == "temporal_link_prediction":
        from .temporal_graph_trainer import temp_graph_trainer as temp_graph_trainer_fn
        return temp_graph_trainer_fn