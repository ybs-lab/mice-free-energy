#params.py

res_params = {
    20 : {'model_config': ('CNNMINE4X_dropout', 20, 2.5 , 0.3, 0.15, 'xavier'),
                'lr': 0.00003, 'ma_rate': 0.0000005, 'batch_size': 500, 'ma_et_start': 1e-5, 'w1': 16, 'w2': 8, 'w3':4, 'w4':2, 'batches': int(1e5)},
    32 : {'model_config': ('CNNMINE4X_dropout', 20, 2.5 , 0.3, 0.15, 'xavier'),
                'lr': 0.00003, 'ma_rate': 0.0000005, 'batch_size': 800, 'ma_et_start': 1e-5, 'w1': 16, 'w2': 8, 'w3':4, 'w4':2, 'batches': int(1e5)},
    36 : {'model_config': ('CNNMINE4X_dropout', 20, 2.5 , 0.3, 0.15, 'xavier'),
                'lr': 0.00003, 'ma_rate': 0.00000025, 'batch_size': 800, 'ma_et_start': 1e-5, 'w1': 20, 'w2': 10, 'w3':6, 'w4':2, 'batches': int(1e5)},
    40 : {'model_config': ('CNNMINE4X_dropout', 22, 2.5 , 0.3, 0.15, 'xavier'),
                'lr': 0.00003, 'ma_rate': 0.00000025, 'batch_size': 1200, 'ma_et_start': 1e-5, 'w1': 20, 'w2': 10, 'w3': 5, 'w4':2, 'batches': int(1e5)},
}


mice_params = {
    (8,8,8) : {'model_config': ('CNNMINE4X_dropout', 20, 2, 0.15, 0.1, 'xavier'),
                'lr': 0.00005, 'ma_rate': 0.00001, 'batch_size': 300, 'ma_et_start': 1e-4, 'w1': 8, 'w2': 4, 'w3':2, 'w4':1, 'batches': int(1.e5)}, 
    (16,16,16) : {'model_config': ('CNNMINE4X_dropout', 20, 2.5, 0.3, 0.15, 'xavier'),
                'lr': 0.00003, 'ma_rate': 0.0000005, 'batch_size': 500, 'ma_et_start': 1e-5, 'w1': 16, 'w2': 8, 'w3':4, 'w4':2, 'batches': int(1e5)},
    (32,32,32) : {'model_config': ('CNNMINE4X_dropout', 20, 2.5 , 0.3, 0.15, 'xavier'),
                'lr': 0.00003, 'ma_rate': 0.0000005, 'batch_size': 800, 'ma_et_start': 1e-5, 'w1': 16, 'w2': 8, 'w3':4, 'w4':2, 'batches': int(1e5)},
    (40,40,40) : {'model_config': ('CNNMINE4X_dropout', 22, 2.5 , 0.3, 0.15, 'xavier'),
                'lr': 0.00003, 'ma_rate': 0.00000025, 'batch_size':1200, 'ma_et_start': 1e-5, 'w1': 20, 'w2': 10, 'w3': 5, 'w4':2, 'batches': int(2e5)}, 
}
