import os
import shutil
import time
import ast

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch.nn.functional as F
import torch.nn as nn

def cls_score(pred, label, prob):
    # organ level
    organ_pred = np.array(pred).ravel()
    organ_true = np.array(label).ravel()
    
    if len(np.unique(organ_true)) < 2:
        organ_auc = 0.0
    else:
        organ_pred_proba = np.array(prob).ravel()
        organ_auc = roc_auc_score(organ_true, organ_pred_proba, multi_class='ovo')

    tn, fp, fn, tp = confusion_matrix(organ_true, organ_pred, labels=[0, 1]).ravel()

    organ_acc = np.sum(organ_pred == organ_true) / (len(organ_true) + 1e-9)
    organ_sen = tp / (tp + fn + 1e-9)
    organ_prec = tp / (tp + fp + 1e-9)
    organ_f1 = (2 * organ_prec * organ_sen) / (organ_prec + organ_sen + 1e-9)

    score_table = {
        "organ_acc": organ_acc,
        "organ_sensitive": organ_sen,
        "organ_precision": organ_prec,
        "organ_f1": organ_f1,
        "organ_auc": organ_auc
    }
    score = 0.3 * organ_auc + 0.2 * organ_f1 + 0.2 * organ_prec + 0.3 * organ_acc

    return score, score_table

def cosine_rampdown(current, rampdown_length):
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def get_ct_rate_text_prompts(labels, text_embeddings):
    batch_prompts = []
    for b in range(labels.shape[0]):
        label_indices = labels[b].nonzero(as_tuple=True)[0]
        if len(label_indices) > 0:
            selected_embeddings = text_embeddings[label_indices]
            prompt = selected_embeddings.mean(dim=0)
        else:
            prompt = torch.zeros_like(text_embeddings[0])
        batch_prompts.append(prompt)
    return torch.stack(batch_prompts)

def train_epoch(model, train_loader, optimizer, scaler, epoch, loss_func, focal_loss_func, args, text_embeddings):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch in enumerate(train_loader):
        data, label_str = batch["img_t1"], batch["label"]
        
        cleaned_labels = ["[" + l.split('[')[-1] for l in label_str]
        parsed_labels = [ast.literal_eval(l) for l in cleaned_labels]
        labels = torch.FloatTensor(parsed_labels).cuda(args.rank)
        data = data.cuda(args.rank)

        with autocast(enabled=args.amp):
            outputs, vl_align_feature = model(data, data)
            ce_loss = loss_func(outputs, labels)
            
            consistency_weight = cosine_rampdown(epoch, args.max_epochs)

            if args.hybrid_loss == "hybrid_loss":
                focal_loss = focal_loss_func(outputs, labels)
                sup_loss = 0.4 * ce_loss + 0.6 * focal_loss
            elif args.hybrid_loss == "focal_loss":
                focal_loss = focal_loss_func(outputs, labels)
                sup_loss = focal_loss
            else:
                sup_loss = ce_loss

            if args.text_prompt_loss:
                text_prompts = get_ct_rate_text_prompts(labels, text_embeddings).cuda(args.rank)
                kl_input = F.log_softmax(vl_align_feature, dim=1)
                kl_target = F.log_softmax(text_prompts, dim=1)
                kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
                KL_prompt_loss = kl_loss(kl_input, kl_target)
            else:
                KL_prompt_loss = 0

            loss = consistency_weight * sup_loss + (1 - consistency_weight) * KL_prompt_loss

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < train_loader.sampler.valid_length)
            run_loss.update(np.mean(loss_list), n=args.batch_size * args.world_size)
        else:
            run_loss.update(loss.item(), n=args.batch_size)

        if args.rank == 0:
            print(
                f"Epoch {epoch}/{args.max_epochs} {idx}/{len(train_loader)}, "
                f"loss: {run_loss.avg:.4f}, "
                f"ce_loss: {ce_loss.item():.4f}, "
                f"KL_prompt_loss: {KL_prompt_loss.item() if isinstance(KL_prompt_loss, torch.Tensor) else KL_prompt_loss:.4f}, "
                f"time {time.time() - start_time:.2f}s"
            )
        start_time = time.time()
    return run_loss.avg

def val_epoch(model, val_loader, epoch, args):
    model.eval()
    y_true = []
    y_pred = []
    y_pred_prob = []
    start_time = time.time()
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            data, label_str = batch["img_t1"], batch["label"]
            
            cleaned_labels = ["[" + l.split('[')[-1] for l in label_str]
            parsed_labels = [ast.literal_eval(l) for l in cleaned_labels]
            labels = torch.FloatTensor(parsed_labels).cuda(args.rank)
            data = data.cuda(args.rank)

            with autocast(enabled=args.amp):
                val_preds, _ = model(data, data)

            y_pred_prob.extend(val_preds.cpu().numpy())
            val_preds[val_preds >= 0.5] = 1
            val_preds[val_preds < 0.5] = 0

            val_labels = labels.cpu().numpy()
            val_preds = val_preds.detach().cpu().numpy()
            y_true.extend(val_labels)
            y_pred.extend(val_preds)

    score, score_table = cls_score(y_pred, y_true, y_pred_prob)
    print(f"Final validation {epoch}/{args.max_epochs - 1}, Score: {score:.4f}, organ_acc: {score_table['organ_acc']:.4f}, organ_auc: {score_table['organ_auc']:.4f}, Organ F1: {score_table['organ_f1']:.4f}, time: {time.time() - start_time:.2f}s")
    return score

def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)

def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    focal_loss_func,
    args,
    scheduler=None,
    start_epoch=0,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        print("Writing Tensorboard logs to", args.logdir)
    
    scaler = GradScaler() if args.amp else None
    val_acc_max = 0.0

    text_embeddings = torch.load(args.CLIP_text_pretrain_dir).cuda(args.rank)

    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()

        print(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, epoch, loss_func, focal_loss_func, args, text_embeddings
        )
        
        if args.rank == 0:
            print(f"Final training {epoch}/{args.max_epochs - 1}, loss: {train_loss:.4f}, time: {time.time() - epoch_time:.2f}s")
            if writer is not None:
                writer.add_scalar("train_loss", train_loss, epoch)
        
        if (epoch + 1) % args.val_every == 0 or (epoch + 1) == args.max_epochs:
            if args.distributed:
                torch.distributed.barrier()

            epoch_time = time.time()
            val_avg_acc = val_epoch(model, val_loader, epoch, args)

            if args.rank == 0:
                print(f"Final validation {epoch}/{args.max_epochs - 1}, acc: {val_avg_acc:.4f}, time: {time.time() - epoch_time:.2f}s")
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print(f"New best ({val_acc_max:.6f} --> {val_avg_acc:.6f}).")
                    val_acc_max = val_avg_acc
                    if args.save_checkpoint:
                        save_checkpoint(model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler)
                if args.save_checkpoint:
                    save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")

        if scheduler is not None:
            scheduler.step()
            
    print("Training Finished! Best Accuracy:", val_acc_max)
    return val_acc_max
