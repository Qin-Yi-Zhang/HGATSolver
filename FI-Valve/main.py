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

def masked_mse_loss(pred, target, mask):
    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    mask = mask.unsqueeze(-1).float()
    loss = ((pred - target) ** 2) * mask
    total_loss = torch.sum(loss)
    num_valid = torch.sum(mask) * pred.shape[1]
    return total_loss / num_valid if num_valid > 0 else torch.tensor(0.0, device=pred.device)

def masked_l2_error(pred, target, mask, relative=False):
    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    mask = mask.unsqueeze(-1).float()
    err_sq = torch.sum(((pred - target) ** 2) * mask)
    if not relative:
        return torch.sqrt(err_sq)
    tgt_sq = torch.sum((target ** 2) * mask)
    if tgt_sq < 1e-12:
        return torch.tensor(0.0, device=pred.device)
    return torch.sqrt(err_sq) / torch.sqrt(tgt_sq)

class IGBLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.log_var_fluid = nn.Parameter(torch.zeros((), device=device))
        self.log_var_solid = nn.Parameter(torch.zeros((), device=device))

    def forward(self, predictions, batch):
        mse_f = masked_mse_loss(predictions.get('fluid', torch.tensor([])), 
                                batch['fluid'].y, batch['fluid'].y_mask) if 'fluid' in predictions else torch.tensor(0.0, device=self.log_var_fluid.device)
        mse_s = masked_mse_loss(predictions.get('solid', torch.tensor([])), 
                                batch['solid'].y, batch['solid'].y_mask) if 'solid' in predictions else torch.tensor(0.0, device=self.log_var_solid.device)

        precision_f = torch.exp(-self.log_var_fluid)
        precision_s = torch.exp(-self.log_var_solid)

        total_loss = 0.5 * (precision_f * mse_f + precision_s * mse_s) + \
                     0.5 * (self.log_var_fluid + self.log_var_solid)
        return total_loss, {'fluid_mse': mse_f, 'solid_mse': mse_s}

def evaluate(model, loader, device, fluid_weight, solid_weight):
    model.eval()
    total_loss = 0.0
    fluid_mses, solid_mses = [], []
    fluid_abs_l2s, solid_abs_l2s = [], []
    fluid_rel_l2s, solid_rel_l2s = [], []

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            batch = batch.to(device)
            predictions = model(batch)

            if 'fluid' in predictions and predictions['fluid'].numel() > 0:
                f_mse = masked_mse_loss(predictions['fluid'], batch['fluid'].y, batch['fluid'].y_mask)
                f_abs = masked_l2_error(predictions['fluid'], batch['fluid'].y, batch['fluid'].y_mask)
                f_rel = masked_l2_error(predictions['fluid'], batch['fluid'].y, batch['fluid'].y_mask, relative=True)
                fluid_mses.append(f_mse.item())
                fluid_abs_l2s.append(f_abs.item())
                fluid_rel_l2s.append(f_rel.item())
            else:
                f_mse = torch.tensor(0.0)

            if 'solid' in predictions and predictions['solid'].numel() > 0:
                s_mse = masked_mse_loss(predictions['solid'], batch['solid'].y, batch['solid'].y_mask)
                s_abs = masked_l2_error(predictions['solid'], batch['solid'].y, batch['solid'].y_mask)
                s_rel = masked_l2_error(predictions['solid'], batch['solid'].y, batch['solid'].y_mask, relative=True)
                solid_mses.append(s_mse.item())
                solid_abs_l2s.append(s_abs.item())
                solid_rel_l2s.append(s_rel.item())
            else:
                s_mse = torch.tensor(0.0)

            total_loss += (fluid_weight * f_mse + solid_weight * s_mse).item()

    n_batches = len(loader)
    if n_batches == 0:
        return {k: 0 for k in ["avg_loss_mse_weighted","avg_fluid_mse","avg_solid_mse",
                               "avg_fluid_abs_l2","avg_solid_abs_l2","avg_fluid_rel_l2","avg_solid_rel_l2"]}

    return {
        "avg_loss_mse_weighted": total_loss / n_batches,
        "avg_fluid_mse": np.mean(fluid_mses) if fluid_mses else 0,
        "avg_solid_mse": np.mean(solid_mses) if solid_mses else 0,
        "avg_fluid_abs_l2": np.mean(fluid_abs_l2s) if fluid_abs_l2s else 0,
        "avg_solid_abs_l2": np.mean(solid_abs_l2s) if solid_abs_l2s else 0,
        "avg_fluid_rel_l2": np.mean(fluid_rel_l2s) if fluid_rel_l2s else 0,
        "avg_solid_rel_l2": np.mean(solid_rel_l2s) if solid_rel_l2s else 0,
    }

def train(args):
    subset_str = f"_data{int(args.data_subset_ratio*100)}pct" if args.data_subset_ratio < 1.0 else ""
    run_name = f"run_{datetime.now().strftime('%y%m%d-%H%M')}_w{args.window_size}_l{args.layers}_h{args.hidden_dim}{subset_str}"
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    train_set = FSIDataset(args.hdf5_path, args.window_size, 'train', data_subset_ratio=args.data_subset_ratio)
    val_set = FSIDataset(args.hdf5_path, args.window_size, 'val')
    test_set = FSIDataset(args.hdf5_path, args.window_size, 'test')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = HGATSolver(
        fluid_input_dim=train_set.fluid_input_dim,
        solid_input_dim=train_set.solid_input_dim,
        fluid_output_dim=train_set.fluid_single_frame_feat_dim,
        solid_output_dim=train_set.solid_single_frame_feat_dim,
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
    history_path = os.path.join(output_dir, 'training_history.json')
    training_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_fn.train()
        total_train_loss = 0
        total_train_fluid_mse = 0
        total_train_solid_mse = 0
        num_batches_processed = 0

        pbar = tqdm(train_loader, desc=f'Train Epoch {epoch}/{args.epochs}', leave=False)
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
            total_train_solid_mse += loss_components['solid_mse'].item()
            num_batches_processed += 1

        if num_batches_processed > 0:
            avg_train_loss = total_train_loss / num_batches_processed
            avg_train_fluid_mse = total_train_fluid_mse / num_batches_processed
            avg_train_solid_mse = total_train_solid_mse / num_batches_processed
        else:
            avg_train_loss = avg_train_fluid_mse = avg_train_solid_mse = 0.0

        val_metrics = evaluate(model, val_loader, device, args.fluid_loss_w, args.solid_loss_w)

        epoch_log = {
            'epoch': epoch,
            'lr': scheduler.get_last_lr()[0],
            'train_loss_uncertainty': avg_train_loss,
            'train_fluid_mse': avg_train_fluid_mse,
            'train_solid_mse': avg_train_solid_mse,
            'log_var_fluid': loss_fn.log_var_fluid.item(),
            'log_var_solid': loss_fn.log_var_solid.item(),
            'precision_fluid': torch.exp(-loss_fn.log_var_fluid).item(),
            'precision_solid': torch.exp(-loss_fn.log_var_solid).item()
        }
        epoch_log.update({f"val_{k}": v for k, v in val_metrics.items()})
        training_history.append(epoch_log)

        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=4)
        
        current_val_loss = val_metrics['avg_loss_mse_weighted']
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save(model.state_dict(), best_model_path)

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        test_metrics = evaluate(model, test_loader, device, args.fluid_loss_w, args.solid_loss_w)
        summary_path = os.path.join(output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(test_metrics, f, indent=4)

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--hdf5_path', type=str, default="PATH_TO_DATA/FI-Valve.hdf5")
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
    p.add_argument('--solid_loss_w', type=float, default=3.0)
    p.add_argument('--output_dir', type=str, default='./output')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--data_subset_ratio', type=float, default=1.0)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    train(args)