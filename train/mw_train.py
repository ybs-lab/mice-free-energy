import time
import argparse
from pathlib import Path

import numpy as np
import torch 
from torch import nn, optim
from torch.utils.data import DataLoader

from ml_colvar import MultiWorkerDataLoad, CNNMINE3X_dropout, CNNMINE2X_dropout, batchTraining, CNNMINE4X_dropout, CNNMINE5X_dropout
from params import res_params, mice_params

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COORDS_DIR = PROJECT_ROOT / "data" / "coordinates"
MODELS_DIR = PROJECT_ROOT / "train" / "models"
RESULTS_BASE_DIR = PROJECT_ROOT / "train" / "results"
LOGS_DIR = PROJECT_ROOT / "train" / "logs"

def get_device():
    """Get the best available device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a CNN on coordinate datasets.')
    parser.add_argument('--run-name', type=str, default=None, help='Identifier for checkpoints and logs')
    parser.add_argument('--seed', type=int, default=12345, help='Seed value')
    parser.add_argument('--bins', type=int, default=32, help='Number of bins per dimension')
    parser.add_argument('--mice', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--limit', type=float, default=0., help='Abort if val MI stays below this after 100k batches')
    parser.add_argument('--dx', type=int, default=32, help='Dimension x (used with --mice)')
    parser.add_argument('--dy', type=int, default=32, help='Dimension y (used with --mice)')
    parser.add_argument('--dz', type=int, default=32, help='Dimension z (used with --mice)')
    parser.add_argument('--element', type=str, default='Na', help='Element tag used in dataset filenames')
    parser.add_argument('--bf', type=float, default=0.3, help='Box fraction used in dataset filenames')
    parser.add_argument('--train-file', type=str, default=None, help='Training dataset filename inside data directory (with or without .npy)')
    parser.add_argument('--val-file', type=str, default=None, help='Validation dataset filename inside data directory (with or without .npy)')
    parser.add_argument('--data-dir', type=str, default=str(DEFAULT_COORDS_DIR), help='Directory containing coordinate tensors')
    parser.add_argument('--nsamples', type=int, default=None, help='Optional limit on the number of samples per split')
    return parser.parse_args()

def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def load_data(nsamples, mice, dx, dy, dz, data_dir, train_filename, val_filename, device):
    data_dir = Path(data_dir)
    train_path = data_dir / f"{train_filename}.npy"
    val_path = data_dir / f"{val_filename}.npy"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing training data: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing validation data: {val_path}")

    # Load data with memory mapping to reduce initial memory usage
    print(f"Loading training data from {train_path}...")
    train_mmap = np.load(train_path, mmap_mode='r')
    print(f"Loading validation data from {val_path}...")
    val_mmap = np.load(val_path, mmap_mode='r')
    
    # Get shapes before processing
    train_shape = train_mmap.shape
    val_shape = val_mmap.shape
    print(f"Training data shape: {train_shape}, Validation data shape: {val_shape}")
    
    # Estimate memory requirements
    train_size_gb = np.prod(train_shape) * 4 / (1024**3)  # float32 = 4 bytes
    val_size_gb = np.prod(val_shape) * 4 / (1024**3)
    total_size_gb = train_size_gb + val_size_gb
    print(f"Estimated memory: {train_size_gb:.2f} GB (train) + {val_size_gb:.2f} GB (val) = {total_size_gb:.2f} GB total")
    print(f"Note: Additional memory will be needed for model, optimizer, and training overhead (~1-2 GB)")
    
    # Determine final shape after slicing
    if nsamples is not None:
        train_shape = (min(nsamples, train_shape[0]),) + train_shape[1:]
        val_shape = (min(nsamples, val_shape[0]),) + val_shape[1:]
    
    if mice:
        train_shape = (train_shape[0], dx, dy, dz)
        val_shape = (val_shape[0], dx, dy, dz)
        print(f"Training MICE with dimensions: {dx}, {dy}, {dz}")
    
    # Apply slicing and convert directly to tensor to minimize memory copies
    print("Processing training data...")
    import gc
    gc.collect()  # Force garbage collection before loading
    
    if nsamples is not None:
        train_slice = train_mmap[:nsamples]
    else:
        train_slice = train_mmap
    
    if mice:
        train_slice = train_slice[:, :dx, :dy, :dz]
    
    # Convert to tensor - use contiguous array to avoid extra copies
    if not train_slice.flags['C_CONTIGUOUS']:
        train_slice = np.ascontiguousarray(train_slice, dtype=np.float32)
    else:
        train_slice = train_slice.astype(np.float32, copy=False)
    
    print("Converting training data to tensor...")
    # torch.from_numpy shares memory, but we need float32
    # On CPU, .to(device) is a no-op, but we call it for consistency
    train_tensor = torch.from_numpy(train_slice)
    if device.type != 'cpu':
        train_tensor = train_tensor.to(device)
    del train_slice, train_mmap  # Free references
    gc.collect()
    
    print("Processing validation data...")
    if nsamples is not None:
        val_slice = val_mmap[:nsamples]
    else:
        val_slice = val_mmap
    
    if mice:
        val_slice = val_slice[:, :dx, :dy, :dz]
    
    # Convert to tensor - use contiguous array to avoid extra copies
    if not val_slice.flags['C_CONTIGUOUS']:
        val_slice = np.ascontiguousarray(val_slice, dtype=np.float32)
    else:
        val_slice = val_slice.astype(np.float32, copy=False)
    
    print("Converting validation data to tensor...")
    val_tensor = torch.from_numpy(val_slice)
    if device.type != 'cpu':
        val_tensor = val_tensor.to(device)
    del val_slice, val_mmap  # Free references
    gc.collect()
    
    # Note: We don't shuffle here - DataLoader with shuffle=True will handle it
    # This saves memory by avoiding an extra copy
    print(f"Loaded {len(train_tensor)} training samples and {len(val_tensor)} validation samples")
    return train_tensor, val_tensor

def _default_dataset_name(split, element, bf, bins):
    return f"coordinates_{split}_{element}_bf{bf}_bin{bins}"

def get_hyperparameters(parser, device):
    hp = {}
    # Handle filename parsing - remove .npy extension if present, but preserve dots in the base name
    if parser.train_file:
        train_stem = parser.train_file
        if train_stem.endswith('.npy'):
            train_stem = train_stem[:-4]
    else:
        train_stem = _default_dataset_name("train", parser.element, parser.bf, parser.bins)
    
    if parser.val_file:
        val_stem = parser.val_file
        if val_stem.endswith('.npy'):
            val_stem = val_stem[:-4]
    else:
        val_stem = _default_dataset_name("val", parser.element, parser.bf, parser.bins)
    
    hp['train_filename'] = train_stem
    hp['val_filename'] = val_stem
    hp['run_name'] = parser.run_name or train_stem
    hp['mi_dim'] = -1
    hp['limit'] = parser.limit
    if parser.mice:
        if parser.dy > parser.dz :
            hp['mi_dim'] = -2
        elif parser.dx > parser.dy :
            hp['mi_dim'] = -3
    hp['log_freq'] = 2000
    hp['in_batch_shuffle'] = False
    
    if not parser.mice:
        for threshold, params in res_params.items():
            if parser.bins <= threshold:
                hp['model_arch'], hp['width'], hp['multiplicity'], hp['dropoutfc'], hp['dropoutconv'], hp['initialization'] = params['model_config']
                hp['lr'], hp['ma_rate'], hp['batch_size'], hp['ma_et_start'] = params['lr'], params['ma_rate'], params['batch_size'], params['ma_et_start']
                hp['batches']=params['batches']
                for w in ['w1', 'w2', 'w3', 'w4', 'w5']:
                    if w in params:
                        hp[w] = params[w]
                if hp['model_arch'] == 'CNNMINE2X_dropout':
                    hp['model'] = CNNMINE2X_dropout(n=hp['width'], k=hp['multiplicity'], 
                                                    w1=hp['w1'], kernel=2, dropoutfc=hp['dropoutfc'], 
                                                    dropoutconv=hp['dropoutconv'], initializations=hp['initialization']).to(device)
                elif hp['model_arch'] == 'CNNMINE3X_dropout':
                    hp['model'] = CNNMINE3X_dropout(n=hp['width'], k=hp['multiplicity'], 
                                                    w1=hp['w1'], w2=hp['w2'], w3=hp['w3'], dropoutfc=hp['dropoutfc'], 
                                                    dropoutconv=hp['dropoutconv'], initializations=hp['initialization']).to(device)
                elif hp['model_arch'] == 'CNNMINE4X_dropout':
                    hp['model'] = CNNMINE4X_dropout(n=hp['width'], k=hp['multiplicity'], 
                                                    w1=hp['w1'], w2=hp['w2'], w3=hp['w3'], w4=hp['w4'], dropoutfc=hp['dropoutfc'], 
                                                    dropoutconv=hp['dropoutconv'], initializations=hp['initialization']).to(device)
                elif hp['model_arch'] == 'CNNMINE5X_dropout':
                    hp['model'] = CNNMINE5X_dropout(n=hp['width'], k=hp['multiplicity'], 
                                                    w1=hp['w1'], w2=hp['w2'], w3=hp['w3'], w4=hp['w4'], w5=hp['w5'], dropoutfc=hp['dropoutfc'], 
                                                    dropoutconv=hp['dropoutconv'], initializations=hp['initialization']).to(device)
                break

    if parser.mice:
        model_created = False
        for threshold, params in mice_params.items():
            if parser.dx <= threshold[0] and parser.dy <= threshold[1] and parser.dz <= threshold[2]:
                hp['model_arch'], hp['width'], hp['multiplicity'], hp['dropoutfc'], hp['dropoutconv'], hp['initialization'] = params['model_config']
                hp['lr'], hp['ma_rate'], hp['batch_size'], hp['ma_et_start'] = params['lr'], params['ma_rate'], params['batch_size'], params['ma_et_start']
                hp['batches']=params['batches']
                for w in ['w1', 'w2', 'w3', 'w4']:
                    if w in params:
                        hp[w] = params[w]   
                if hp['model_arch'] == 'CNNMINE2X_dropout':
                    hp['model'] = CNNMINE2X_dropout(n=hp['width'], k=hp['multiplicity'], 
                                                    w1=hp['w1'], kernel=2, dropoutfc=hp['dropoutfc'], 
                                                    dropoutconv=hp['dropoutconv'], initializations=hp['initialization']).to(device)
                elif hp['model_arch'] == 'CNNMINE3X_dropout':
                    hp['model'] = CNNMINE3X_dropout(n=hp['width'], k=hp['multiplicity'], 
                                                    w1=hp['w1'], w2=hp['w2'], w3=hp['w3'], dropoutfc=hp['dropoutfc'], 
                                                    dropoutconv=hp['dropoutconv'], initializations=hp['initialization']).to(device)
                elif hp['model_arch'] == 'CNNMINE4X_dropout':
                    hp['model'] = CNNMINE4X_dropout(n=hp['width'], k=hp['multiplicity'], 
                                                    w1=hp['w1'], w2=hp['w2'], w3=hp['w3'], w4=hp['w4'], dropoutfc=hp['dropoutfc'], 
                                                    dropoutconv=hp['dropoutconv'], initializations=hp['initialization']).to(device)
                model_created = True
                break
        
        if not model_created:
            available_thresholds = list(mice_params.keys())
            raise ValueError(
                f"No model configuration found for MICE dimensions ({parser.dx}, {parser.dy}, {parser.dz}).\n"
                f"Available configurations: {available_thresholds}\n"
                f"Note: Dimensions must be <= threshold values. For example, (16,16,16) requires a threshold of at least (16,16,16)."
            )
    
    if 'model' not in hp:
        raise ValueError("Model was not created. Check your parameters and ensure a matching configuration exists.")
    
    print(hp['model'])
    pytorch_total_params = sum(p.numel() for p in hp['model'].parameters())
    print("number of parameters: ", pytorch_total_params)
    print(f"training exp {hp['train_filename']}")
    print(f"For {hp['batches']} batches, {hp['lr']} learning rate, {hp['ma_rate']} moving average rate")
    return hp
           
def main():
    device = get_device()
    print(f"Using device: {device}")
    hp = get_hyperparameters(parser, device)
    
    # Create separate results directories for MICE and regular (res) results
    if parser.mice:
        RESULTS_DIR = RESULTS_BASE_DIR / "mice"
    else:
        RESULTS_DIR = RESULTS_BASE_DIR / "res"
    
    # Create all necessary directories (base dir and subdirectories)
    RESULTS_BASE_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_BASE_DIR / "mice").mkdir(parents=True, exist_ok=True)
    (RESULTS_BASE_DIR / "res").mkdir(parents=True, exist_ok=True)
    
    for path in (MODELS_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)
    train_data, val_data = load_data(parser.nsamples, parser.mice, 
                                     parser.dx, parser.dy, parser.dz,
                                     parser.data_dir, hp['train_filename'], hp['val_filename'], device)
    
    optimizer = optim.Adam(hp['model'].parameters(), lr=hp['lr'])
    earlyStopper = None
    if earlyStopper is not None:
        print("early stopping enabled")
        
    trainData = MultiWorkerDataLoad(train_data, inBatchShuffle=hp['in_batch_shuffle'])
    valData = MultiWorkerDataLoad(val_data, inBatchShuffle=hp['in_batch_shuffle'])

    trainDataLoader = DataLoader(trainData, batch_size=hp['batch_size'],
                                 collate_fn=lambda batch: trainData.my_collate(batch,dim=hp['mi_dim']), shuffle=True)
    valDataLoader = DataLoader(valData, batch_size=hp['batch_size'],
                               collate_fn=lambda batch: valData.my_collate(batch,dim=hp['mi_dim']), shuffle=True)
    print("number of train batches: ", len(trainDataLoader))    
    print("number of val batches: ", len(valDataLoader))

    if parser.mice:
        hp['run_name'] += f'_mice_dx{parser.dx}_dy{parser.dy}_dz{parser.dz}'
    hp['run_name'] += f'_s{parser.seed}'
    run_prefix = hp['run_name']
    train_filename = f"{run_prefix}_w{hp['w1']}_b{hp['batches']}_lr{hp['lr']}_ma{hp['ma_rate']}_bs{hp['batch_size']}_width{hp['width']}_m{hp['multiplicity']}_dfc{hp['dropoutfc']}_dconv{hp['dropoutconv']}_init{hp['initialization']}"
    model_filename = MODELS_DIR / f"{train_filename}_s{parser.seed}_model.pt"
    ts, vs, loss, Et, logEet, ma_et_l = batchTraining(trainDataLoader, valDataLoader, hp['model'], 
                                                      optimizer, hp['batches'], hp['log_freq'], hp['ma_rate'], 
                                                      stable=True, unbiased=True, model_filename=model_filename, 
                                                      ma_et_start=hp['ma_et_start'], limit=0., device=device)

    ts = np.array(ts).reshape(-1)
    vs = np.array(vs).reshape(-1)
    loss = np.array(loss).reshape(-1)
    Et = np.array(Et).reshape(-1)
    logEet = np.array(logEet).reshape(-1)
    ma_et_l = np.array(ma_et_l).reshape(-1)
    metrics = np.array([ts, vs, loss, Et, logEet, ma_et_l], dtype=object)
    results_path = RESULTS_DIR / f"{train_filename}_s{parser.seed}_metrics.npy"
    np.save(results_path, metrics)
    log_path = LOGS_DIR / f"{train_filename}_s{parser.seed}.log"
    with open(log_path, "w", encoding="utf-8") as logfile:
        logfile.write(f"run_name={run_prefix}\n")
        logfile.write(f"train_file={hp['train_filename']}.npy\n")
        logfile.write(f"val_file={hp['val_filename']}.npy\n")
        logfile.write(f"seed={parser.seed}\n")
        logfile.write(f"batch_size={hp['batch_size']}\n")
        logfile.write(f"batches={hp['batches']}\n")
        logfile.write(f"lr={hp['lr']}\n")
        logfile.write(f"ma_rate={hp['ma_rate']}\n")
        logfile.write(f"ts_final={ts[-1] if len(ts) else 'n/a'}\n")
        logfile.write(f"vs_final={vs[-1] if len(vs) else 'n/a'}\n")
        logfile.write(f"metrics_file={results_path}\n")
        logfile.write(f"model_checkpoint={model_filename}\n")
    print(f"finished training {train_filename}")
    print(f"params: w1: {hp['w1']}, w2: {hp['w2']}, w3: {hp['w3']}, w4: {hp['w4']}, batches: {hp['batches']}, batch_size: {hp['batch_size']}, lr: {hp['lr']}, ma_rate: {hp['ma_rate']}, width: {hp['width']}, multiplicity: {hp['multiplicity']}, dropoutfc: {hp['dropoutfc']}, dropoutconv: {hp['dropoutconv']}, initialization: {hp['initialization']}")
    print("time taken: ", time.time()-time_start)

parser = parse_arguments()
time_start=time.time()
init_seed(parser.seed)

if __name__=="__main__":
    main()