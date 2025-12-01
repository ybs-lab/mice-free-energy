import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from time import time
from torch.utils.checkpoint import checkpoint

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_loss = 0

    def early_stop(self, validation_loss):
        if np.isnan(validation_loss):
            print("early stopping due to nan")
            return True
        if validation_loss > self.max_validation_loss:
            self.max_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss < (self.max_validation_loss*(1-self.min_delta)):
            self.counter += 1
            if self.counter >= self.patience:
                print("early stopping")
                return True
        return False
    
class DefaultDataLoad:
    def __init__(self, data):
        self.data = data
        self.len = self.data.shape[0]
        self.k = int(self.data.shape[-1]/2)
        self.indices = np.arange(self.len)

    def getBatch(self, batchsize):
        if batchsize > len(self.indices):
            self.indices = np.arange(self.len)
            # print("finished epoch")
        idx = np.random.choice(self.indices, size=(2, batchsize), replace=False)
        self.indices = np.setdiff1d(self.indices, idx)
        marginal = np.concatenate(
            (self.data[idx[0]][..., :self.k], self.data[idx[1]][..., self.k:]), axis=-1)
        return np.stack((self.data[idx[0]], marginal), axis=1)

  
class MultiWorkerDataLoad(Dataset):
    def __init__(self, data, inBatchShuffle=False):
        self.data = data.float() if isinstance(data, torch.Tensor) else torch.from_numpy(data).float()
        self.k = int(data.shape[-1]/2)
        self.inBatchShuffle = inBatchShuffle

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

    def my_collate(self, batch, dim=-1):
        t = batch if isinstance(batch, torch.Tensor) else torch.stack(batch)
        k = int(t.shape[dim]/2)  # Split based on the specified dimension
        batch_size = t.size(0)
        
        if self.inBatchShuffle:
            indices_Y = torch.randperm(batch_size)
            marginal = torch.cat((t.index_select(dim, torch.arange(k)),
                                  t.index_select(dim, indices_Y).index_select(dim, torch.arange(k, 2*k))), dim=dim)
        else:
            indices_Y = torch.randint(0, self.__len__(), (batch_size,), device=t.device)
            marginal = torch.cat((t.index_select(dim, torch.arange(k, device=t.device)),
                                  self.data.index_select(0, indices_Y).index_select(dim, torch.arange(k, 2*k,device=t.device))), dim=dim)

        return torch.stack((t, marginal), dim=1)


class CNNMINE4X_dropout(nn.Module):
    def __init__(self, n=20, k=2.5, w1=16, w2=8, w3=4, w4=2, dropoutfc=0.3, dropoutconv=0.15, input_size=1, initializations=None):
        super().__init__()
        self.conv1 = nn.Conv3d(input_size, n, kernel_size=4, stride=1, padding=1)  # Retain more spatial size
        self.conv2 = nn.Conv3d(n, int(k*n), kernel_size=4, stride=1, padding=1)   # Gradual reduction with stride 2
        self.conv3 = nn.Conv3d(int(k*n), int(n*k*k), kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(int(n*k*k), int(n*k*k*k/2), kernel_size=3, stride=1, padding=1)  # Use stride for gradual reduction

        # Adaptive max pooling with gradual reduction in target sizes
        self.adaptivemaxpool3d_1 = nn.AdaptiveMaxPool3d(w1)
        self.adaptivemaxpool3d_2 = nn.AdaptiveMaxPool3d(w2)
        self.adaptivemaxpool3d_3 = nn.AdaptiveMaxPool3d(w3)
        self.adaptivemaxpool3d_4 = nn.AdaptiveMaxPool3d(w4)

        self.fc1 = nn.Linear(int(n*k*k*k/2) * w4**3, int(k*n))  # Adjust for flattened size
        self.fc2 = nn.Linear(int(k*n), int(k*n/2))
        self.fc3 = nn.Linear(int(k*n/2), n)
        self.fc4 = nn.Linear(n, 1)

        self.relu = nn.LeakyReLU()
        self.fcdropout = nn.Dropout(p=dropoutfc)
        self.convdropout = nn.Dropout3d(p=dropoutconv)

        # Weight initialization
        if initializations is not None:
            initializer = None
            if initializations == 'xavier':
                initializer = torch.nn.init.xavier_uniform_
            elif initializations == 'kaiming':
                initializer = torch.nn.init.kaiming_uniform_
            if initializer:
                initializer(self.conv1.weight)
                initializer(self.conv2.weight)
                initializer(self.conv3.weight)
                initializer(self.conv4.weight)
                initializer(self.fc1.weight)
                initializer(self.fc2.weight)
                initializer(self.fc3.weight)
                initializer(self.fc4.weight)

    def forward(self, x):
        # Convolutional and pooling layers with gradual pooling
        out = self.conv1(x)
        out = self.relu(out)
        out = self.convdropout(out)
        out = self.adaptivemaxpool3d_1(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.convdropout(out)
        out = self.adaptivemaxpool3d_2(out)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.convdropout(out)
        out = self.adaptivemaxpool3d_3(out)

        out = self.conv4(out)
        out = self.relu(out)
        out = self.convdropout(out)
        out = self.adaptivemaxpool3d_4(out)

        # Flatten for fully connected layers
        out = out.view(out.size(0), -1)

        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fcdropout(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.fcdropout(out)

        out = self.fc3(out)
        out = self.relu(out)
        out = self.fcdropout(out)

        out = self.fc4(out)
        return out


def trainBatch(epoch, batch, model, optimizer, ma_et, ma_rate, unbiased=True, stable=False, mixed_precision=False):
    model.train()
    optimizer.zero_grad()
    joint, marginal = torch.split(batch, 1, dim=1)
    # print(torch.cuda.memory_summary())
    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()     
        with torch.cuda.amp.autocast():
            t = model(joint)
    else:
        t = model(joint)

    # print(torch.cuda.memory_summary())

    if stable:
        if mixed_precision:
            with torch.cuda.amp.autocast():
                t_marginal = model(marginal)
                # print(torch.cuda.memory_summary())
                log_sum_exp_et = torch.logsumexp(t_marginal, 0)
                et = torch.exp(log_sum_exp_et - torch.log(torch.tensor(t_marginal.shape[0])))
                Et = torch.mean(t)
                Eet = torch.mean(et)
                logEet = torch.log(Eet)
                mi_lb = Et - logEet
                ma_et = (1 - ma_rate) * ma_et + ma_rate * Eet.detach()
                if unbiased:
                    loss = -(Et - (1 / (ma_et + 1e-6)).detach() * Eet)
                else:
                    loss = -mi_lb
        else:
            t_marginal = model(marginal)
            log_sum_exp_et = torch.logsumexp(t_marginal, 0)
            et = torch.exp(log_sum_exp_et - torch.log(torch.tensor(t_marginal.shape[0])))
            Et = torch.mean(t)
            Eet = torch.mean(et)
            logEet = torch.log(Eet)
            mi_lb = Et - logEet
            ma_et = (1 - ma_rate) * ma_et + ma_rate * Eet
            if unbiased:
                loss = -(Et - (1 / (ma_et + 1e-6)).detach() * Eet)
            else:
                loss = -mi_lb

    else:
        t_marginal = model(marginal)
        et = torch.exp(t_marginal)
        Et = torch.mean(t)
        Eet = torch.mean(et)
        logEet = torch.log(Eet)
        mi_lb = Et - logEet
        ma_et = (1-ma_rate)*ma_et + ma_rate*Eet
        if unbiased:
            loss = -(Et - (1/ma_et).detach()*Eet) #+ torch.max(torch.tensor(0.0), Et)
        else:
            loss = - mi_lb
    
    if mixed_precision and torch.cuda.is_available():
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.empty_cache()
    else:
        loss.backward()
        optimizer.step()
    
    # loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm='inf')
    # plot_grad_flow(model.named_parameters())
    # optimizer.step()
    
    return mi_lb.detach().to(torch.device('cpu')).numpy(), ma_et.detach().to(torch.device('cpu')).numpy(), loss.detach().to(torch.device('cpu')).numpy(), Et.detach().to(torch.device('cpu')).numpy(), logEet.detach().to(torch.device('cpu')).numpy(), Eet.detach().to(torch.device('cpu')).numpy()


def valBatch(batch, model,stable=False):
    model.eval()
    with torch.no_grad():
        joint, marginal = torch.split(batch, 1, dim=1)
        # joint, marginal = batch
        t = model(joint)
        if stable:
            t_marginal = model(marginal)
            logsumet = torch.logsumexp(t_marginal, 0) - torch.log(torch.tensor(marginal.shape[0]))
            mi_lb = (torch.mean(t) - logsumet)
        else:
            et = torch.exp(model(marginal))
            mi_lb = torch.mean(t) - torch.log(torch.mean(et))

    return mi_lb.detach().to(torch.device('cpu')).numpy()


def batchEvaluation(dataLoader, model, batches, log_freq=-1, stable=False):
    model.eval()
    valMIlb = []
    for batch in range(batches):
        vBatch = next(iter(dataLoader))
        val_mi_lb = valBatch(vBatch, model, stable=stable)
        valMIlb.append(val_mi_lb)

        if log_freq > 0:
                if (batch) % (log_freq) == 0:
                    print(f'batch {batch} | Val MILB: {valMIlb[-1]}')
    
    return valMIlb

def batchTraining(trainDataLoader, valDataLoader, model, optimizer, 
                     batches=3000, log_freq=-1, ma_rate=0.001, 
                     unbiased=True, stable=False, model_filename=None,
                     ma_et_start=1., limit=0, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    trainMIlb, valMIlb = [], []
    ma_et_l, loss_l, Et_l, logEet_l, Eet_l = [], [], [], [], []
    maxMI=0
    ma_et = ma_et_start
    
    for batch in range(batches):    
        tBatch = next(iter(trainDataLoader))
        tBatch = tBatch.to(device)
        train_mi_lb, ma_et, loss, Et, logEet, Eet = trainBatch(batch, tBatch, model, optimizer, ma_et, ma_rate, unbiased, stable)
        trainMIlb.append(train_mi_lb)
        ma_et_l.append(ma_et)
        loss_l.append(loss)
        Et_l.append(Et)
        logEet_l.append(logEet)
        Eet_l.append(Eet)
        
        # Free GPU memory for training batch
        # del tBatch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        vBatch = next(iter(valDataLoader))
        vBatch = vBatch.to(device)
        val_mi_lb = valBatch(vBatch, model, unbiased)
        valMIlb.append(val_mi_lb)
        
        # Free GPU memory for validation batch
        # del vBatch
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        if batch == 100000 and np.mean(valMIlb[-100:]) < limit:
            print(f"aborting training due to insufficient MI, MI: {np.mean(valMIlb[-100:])}")
            break
    
        if log_freq > 0:
            if batch % log_freq == 0:
                print(f'b {batch} | TMI: {np.mean(trainMIlb[-1]):.6f} std: {np.std(trainMIlb[-100:]):.6f} | ' 
                    f'VMI: {np.mean(valMIlb[-100:]):.6f} std: {np.std(valMIlb[-100:]):.6f} | '
                    f'Loss: {np.mean(loss_l[-100:]):.6f} | Et: {np.mean(Et_l[-100:]):.6f} | '
                    f'logEet: {np.mean(logEet_l[-100:]):.6f} | Eet: {np.mean(Eet_l[-100:]):.6f} | '
                    f'ma_et: {np.mean(ma_et_l[-100:]):.6f}')
                if np.mean(valMIlb[-100:]) > maxMI and batch > batches/2: 
                    maxMI = np.mean(valMIlb[-100:])
                    if model_filename is not None:
                        from pathlib import Path
                        # Handle both Path objects and string paths
                        model_path = Path(model_filename)
                        model_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(model_path, "wb") as f:
                            torch.save(model.state_dict(), f)
                if np.isnan(np.mean(valMIlb[-100:])):
                    print("early stopping due to nan")
                    break
        

                            
    return trainMIlb, valMIlb, loss_l, Et_l, logEet_l, ma_et_l

def multiWorkerTrain(trainDataLoader, valDataLoader, model, optimizer, earlyStopper,
                     epochs=3000, batch_size=32, log_freq=-1, ma_rate=0.001, 
                     unbiased=True, timing=False, stable=False, limit=0):
    
    # autograd.set_detect_anomaly(True)
    
    trainMIlb, valMIlb = [], []
    num_of_train_batches = len(trainDataLoader)
    num_of_val_batches = len(valDataLoader)

    ma_et = 1.e-4
    for epoch in range(epochs):

        if timing:
            time0 = time()

        train_ep, val_ep = 0, 0
        for tBatch in trainDataLoader:
            train_mi_lb, ma_et = trainBatch(
                epoch, tBatch, model, optimizer, ma_et, ma_rate, unbiased, stable, limit)
            train_ep += train_mi_lb
        for vBatch in valDataLoader:
            val_mi_lb = valBatch(vBatch, model, unbiased)
            val_ep += val_mi_lb

        trainMIlb.append(train_ep/num_of_train_batches)
        valMIlb.append(val_ep/num_of_val_batches)
        
        if earlyStopper is not None:
            if earlyStopper.early_stop(val_ep):          
                break
        
        if timing:
            print("time taken for epoch: ", time()-time0)

        if log_freq > 0:
            if (epoch) % (log_freq) == 0:
                print(f'epoch {epoch} | Train MILB: {trainMIlb[-1]} Val MILB: {valMIlb[-1]}')

    return trainMIlb, valMIlb


def ma(a, window_size=100):
    start = [np.mean(a[0:i]) for i in range(0, window_size)]
    middle = [np.mean(a[i:i+window_size])
              for i in range(window_size, len(a)-window_size)]
    end = [np.mean(a[i:len(a)]) for i in range(len(a)-window_size, len(a))]
    return np.concatenate((np.array(start), np.array(middle), np.array(end)))