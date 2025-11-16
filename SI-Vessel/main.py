import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import random
import numpy as np
from datetime import datetime
import json

from dataset import FSIDataset
from model import HGATSolver

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def mse_loss(pred, target):
    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    return torch.mean((pred - target) ** 2)

def l2_error(pred, target, relative=False):
    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    error_sq = torch.sum((pred - target) ** 2)
    if not relative:
        return torch.sqrt(error_sq)
    target_sq = torch.sum(target ** 2)
    if target_sq < 1e-12:
        return torch.tensor(0.0, device=pred.device)
    return torch.sqrt(error_sq) / torch.sqrt(target_sq)

class IGBLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.log_var_fluid = nn.Parameter(torch.zeros((), device=device))
        self.log_var_solid_elastic = nn.Parameter(torch.zeros((), device=device))

    def forward(self, predictions, batch):
        loss_components = {}
        mse_f = mse_loss(predictions['fluid'], batch['fluid'].y) if 'fluid' in predictions else torch.tensor(0.0, device=self.log_var_fluid.device)
        mse_se = mse_loss(predictions['solid_elastic'], batch['solid_elastic'].y) if 'solid_elastic' in predictions else torch.tensor(0.0, device=self.log_var_solid_elastic.device)
        loss_components['fluid_mse'] = mse_f
        loss_components['solid_elastic_mse'] = mse_se
        precision_f = torch.exp(-self.log_var_fluid)
        precision_se = torch.exp(-self.log_var_solid_elastic)
        total_loss = 0.5 * (precision_f * mse_f + precision_se * mse_se) + 0.5 * (self.log_var_fluid + self.log_var_solid_elastic)
        return total_loss, loss_components

def evaluate(model, loader, device, fluid_weight, solid_elastic_weight):
    model.eval()
    total_loss_mse_weighted = 0.0
    fluid_mses, solid_elastic_mses = [], []
    fluid_abs_l2s, solid_elastic_abs_l2s = [], []
    fluid_rel_l2s, solid_elastic_rel_l2s = [], []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            batch = batch.to(device)
            predictions = model(batch)

            if 'fluid' in predictions and predictions['fluid'].numel() > 0:
                fluid_mse = mse_loss(predictions['fluid'], batch['fluid'].y)
                fluid_abs_l2 = l2_error(predictions['fluid'], batch['fluid'].y, relative=False)
                fluid_rel_l2 = l2_error(predictions['fluid'], batch['fluid'].y, relative=True)
                fluid_mses.append(fluid_mse.item())
                fluid_abs_l2s.append(fluid_abs_l2.item())
                fluid_rel_l2s.append(fluid_rel_l2.item())
            else:
                fluid_mse = torch.tensor(0.0)

            if 'solid_elastic' in predictions and predictions['solid_elastic'].numel() > 0:
                solid_elastic_mse = mse_loss(predictions['solid_elastic'], batch['solid_elastic'].y)
                solid_elastic_abs_l2 = l2_error(predictions['solid_elastic'], batch['solid_elastic'].y, relative=False)
                solid_elastic_rel_l2 = l2_error(predictions['solid_elastic'], batch['solid_elastic'].y, relative=True)
                solid_elastic_mses.append(solid_elastic_mse.item())
                solid_elastic_abs_l2s.append(solid_elastic_abs_l2.item())
                solid_elastic_rel_l2s.append(solid_elastic_rel_l2.item())
            else:
                solid_elastic_mse = torch.tensor(0.0)
            
            total_loss_mse_weighted += (fluid_weight * fluid_mse + solid_elastic_weight * solid_elastic_mse).item()

    num_batches = len(loader)
    if num_batches == 0:
        return {"avg_loss_mse_weighted": 0}
    metrics = {
        "avg_loss_mse_weighted": total_loss_mse_weighted / num_batches,
        "avg_fluid_mse": np.mean(fluid_mses) if fluid_mses else 0,
        "avg_solid_elastic_mse": np.mean(solid_elastic_mses) if solid_elastic_mses else 0,
        "avg_fluid_abs_l2": np.mean(fluid_abs_l2s) if fluid_abs_l2s else 0,
        "avg_solid_elastic_abs_l2": np.mean(solid_elastic_abs_l2s) if solid_elastic_abs_l2s else 0,
        "avg_fluid_rel_l2": np.mean(fluid_rel_l2s) if fluid_rel_l2s else 0,
        "avg_solid_elastic_rel_l2": np.mean(solid_elastic_rel_l2s) if solid_elastic_rel_l2s else 0,
    }
    return metrics

def train(args):
    run_name = f"run_{datetime.now().strftime('%y%m%d-%H%M')}_w{args.window_size}_l{args.layers}_h{args.hidden_dim}"
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    print(f"device: {device}")
    
    train_set = FSIDataset(args.hdf5_path, args.window_size, 'train')
    val_set = FSIDataset(args.hdf5_path, args.window_size, 'val')
    test_set = FSIDataset(args.hdf5_path, args.window_size, 'test')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = HGATSolver(
        fluid_input_dim=train_set.fluid_input_dim,
        solid_elastic_input_dim=train_set.solid_elastic_input_dim,
        fluid_output_dim=train_set.fluid_single_frame_feat_dim,
        solid_elastic_output_dim=train_set.solid_elastic_single_frame_feat_dim,
        phys_params_dim=train_set.phys_params_dim,
        coord_dim=args.coord_dim,
        time_emb_dim=args.time_emb_dim,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout
    ).to(device)

    loss_fn = IGBLoss(device)
    optimizer = optim.AdamW([
        {'params': model.parameters()},
        {'params': loss_fn.parameters(), 'lr': args.lr * 10}
    ], lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

    best_val_loss = float('inf')
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, 'best_model.pt')
    training_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_fn.train()
        total_train_loss, total_train_fluid_mse, total_train_se_mse = 0, 0, 0
        pbar = tqdm(train_loader, desc=f'epochs {epoch}/{args.epochs}', leave=False)
        for batch in pbar:
            if batch is None:
                continue
            batch = batch.to(device)
            optimizer.zero_grad()
            predictions = model(batch)
            loss, loss_components = loss_fn(predictions, batch)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()
            total_train_fluid_mse += loss_components['fluid_mse'].item()
            total_train_se_mse += loss_components['solid_elastic_mse'].item()
        
        num_train_batches = len(train_loader)
        avg_train_loss = total_train_loss / num_train_batches
        avg_train_fluid_mse = total_train_fluid_mse / num_train_batches
        avg_train_se_mse = total_train_se_mse / num_train_batches
        
        val_metrics = evaluate(model, val_loader, device, args.fluid_loss_w, args.solid_elastic_loss_w)
        epoch_log = {
            'epoch': epoch, 'lr': scheduler.get_last_lr()[0],
            'train_loss_uncertainty': avg_train_loss,
            'train_fluid_mse': avg_train_fluid_mse,
            'train_solid_elastic_mse': avg_train_se_mse,
        }
        epoch_log.update({f"val_{k}": v for k, v in val_metrics.items()})
        training_history.append(epoch_log)

        current_val_loss = val_metrics['avg_loss_mse_weighted']
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save(model.state_dict(), best_model_path)

    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=4)

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        test_metrics = evaluate(model, test_loader, device, args.fluid_loss_w, args.solid_elastic_loss_w)
        summary_path = os.path.join(output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(test_metrics, f, indent=4)

def parse_arguments():
    p = argparse.ArgumentParser(description="Training script")
    p.add_argument('--hdf5_path', type=str, default="PATH_TO_DATA/SI-Vessel.hdf5")
    p.add_argument('--output_dir', type=str, default='outputs')
    p.add_argument('--window_size', type=int, default=10)
    p.add_argument('--coord_dim', type=int, default=2)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--time_emb_dim', type=int, default=32)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--wd', type=float, default=1e-5)
    p.add_argument('--clip_grad_norm', type=float, default=1.0)
    p.add_argument('--fluid_loss_w', type=float, default=1.0)
    p.add_argument('--solid_elastic_loss_w', type=float, default=3.0)
    p.add_argument('--num_workers', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda:0')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    train(args)