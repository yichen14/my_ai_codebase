
def dispatcher(cfg):
    task_name = cfg.task
    assert task_name != "none"
    if task_name == "graph_link_prediction":
        from .gae_trainer import gae_trainer as gae_trainer_fn
        return gae_trainer_fn