
def dispatcher(cfg):
    task_name = cfg.task
    assert task_name != "none"
    if task_name == "graph_link_prediction":
        from .autoencoder_trainer import autoencoder_trainer as autoencoder_trainer_fn
        return autoencoder_trainer_fn