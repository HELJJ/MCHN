hyperparameter_defaults = dict(
    dataset='mimic3',
    task='m',
    batch_size=32,
    lstm_num_layers=2,
    seed=6669,
    epochs = 200
    )
sweep_config = {
    "name": "hypergraph_soft_attention",
    "metric": {"name": "f1_score", "goal": "maximize"},
    "method": "grid",
    "parameters": {
        "hidden_size": {
            "values": [150, 256, 512]
        },
        "code_size": {
            "values": [48, 128, 256, 512]
        },
        "graph_size": {
            "values": [32, 128, 256, 512]
        },
        "t_attention_size": {
            "values": [32, 128, 512]
        }
    }
}