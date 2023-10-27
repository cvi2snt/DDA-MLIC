import os
import argparse
import time

import torch
from torch.optim import lr_scheduler, Adam
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from randaugment import RandAugment

from src.helper_functions.utils import get_dataset_mlic
from src.helper_functions.helper_functions import mAP, add_weight_decay, ForeverDataIterator, CutoutPIL
from src.helper_functions.logger import CompleteLogger
from src.helper_functions.meter import AverageMeter, ProgressMeter
from src.models import create_model
from src.loss_functions.losses import AsymmetricLoss, GMM_Discrepancy

parser = argparse.ArgumentParser(description='PyTorch TResNet Discriminator-free Domain Adaptation Training')
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--model-name', default='tresnet_m')
parser.add_argument('--logs-dir', default='runs/')
parser.add_argument('-s', '--source', default='aid')
parser.add_argument('-t', '--target', default='ucm')
parser.add_argument('-s-dir', '--source-dir', default='datasets/AID')
parser.add_argument('-t-dir', '--target-dir', default='datasets/UCM')
parser.add_argument('--phase', default='train')
parser.add_argument('--model-path', default=None, type=str)
parser.add_argument('--num-classes', default=17)
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=224, type=int,
                    metavar='N', help='input image size (default: 224)')
parser.add_argument('--thre', default=0.8, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 64)')
parser.add_argument('-i', '--iters-per-epoch', default=100, type=int,
                        help='Number of iterations per epoch')
parser.add_argument('--reg-0', default=1.0, type=float,
                    metavar='N', help='Regularizer value for Gaussian of 0s')
parser.add_argument('--reg-1', default=1.0, type=float,
                    metavar='N', help='Regularizer value for Gaussian of 1s')
parser.add_argument('--trade-off', default=1.0, type=float,
                    metavar='N', help='Regularizer value for discrepancy')

def main():
    args = parser.parse_args()
    args.do_bottleneck_head = False
    logger = CompleteLogger(os.path.join(args.logs_dir, args.source+'2'+args.target), args.phase)
    print(args)    
    
    # Data loading
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        # normalize, # no need, toTensor does normalization
    ])
    train_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            CutoutPIL(cutout_factor=0.5),
            RandAugment(),
            transforms.ToTensor(),
            # normalize,
        ])
    
    s_train_dataset, s_val_dataset, t_train_dataset, t_val_dataset, args.num_classes, class_names = get_dataset_mlic(
        source=args.source, target=args.target, 
        source_dir=args.source_dir, target_dir=args.target_dir, 
        train_transform=train_transform, val_transform=val_transform)
    
    s_train_loader = DataLoader(s_train_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    t_train_loader = DataLoader(t_train_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    s_val_loader = DataLoader(s_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    t_val_loader = DataLoader(t_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    
    
    print( 'len(s_train_loader), len(s_val_loader)' ,len(s_train_loader), len(s_val_loader))
    print( 'len(t_train_loader), len(t_val_loader)' ,len(t_train_loader), len(t_val_loader))
    
    # Setup model
    print('--> Creating model...')
    model = create_model(args).cuda()    
    print('Model creation complete\n')   
    
    ##### Testing
    if args.phase == 'test':
        model = load_model_weights_finetune(model, args.model_path)        
        print("Target {} Validation in progress".format(args.target))
        validate_full(t_val_loader, model, args)
        return
    
    ##### Training   
    # Loading pre-trained model
    if args.model_path:
        model = load_model_weights_pretrain(model, args.model_path)
    
    train(model, s_train_loader, t_train_loader, args, s_val_loader, t_val_loader, logger)
    
    print('\n------Final Validation------------')
    model = load_model_weights_finetune(model, logger.get_checkpoint_path('best'))
    model.eval()        
    print("Target Validation in progress")
    validate_full(t_val_loader, model, args)    
    logger.close()
           
def train(model, dataloader_source, dataloader_target, args, s_val_loader, t_val_loader, logger):
    
    # set optimizer
    Epochs = 25
    weight_decay = 1e-4
    args.iters_per_epoch = max(len(dataloader_source), len(dataloader_target))    
    parameters = add_weight_decay(model, weight_decay) # true wd, filter_bias_and_bn
    optimizer = Adam(params=parameters, lr=args.lr, weight_decay=0) 
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=args.iters_per_epoch, epochs=Epochs,
                                        pct_start=0.2)
    # loss functions
    asl = AsymmetricLoss(gamma_neg=3, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True, eps=1e-6)     
    # Classifier as discriminator (Disc. Loss)
    discrepancy = GMM_Discrepancy(model.head, args)  

    train_source_iter = ForeverDataIterator(dataloader_source)
    train_target_iter = ForeverDataIterator(dataloader_target)
    
    highest_mAP_t = 0
    highest_mAP_s = 0

    scaler = GradScaler()
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    ASL = AverageMeter('Loss', ':3.2f')
    Disc = AverageMeter('Disc Loss', ':5.4f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    tgt_accs = AverageMeter('Tgt Acc', ':3.1f')

    for epoch in range(Epochs):
        progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, ASL, Disc, cls_accs, tgt_accs],
        prefix="Epoch: [{}]".format(epoch))

        model.train()
        discrepancy.train()
        end = time.time()
        for i in range(args.iters_per_epoch):
            x_s, labels_s = next(train_source_iter)[:2]
            x_t, labels_t = next(train_target_iter)[:2]
            if args.source=='coco':
                labels_s, labels_t = labels_s.max(dim=1)[0],labels_t.max(dim=1)[0]  

            # same sample size in source and target
            B = min(x_s.shape[0], x_t.shape[0]) 
            x_s, labels_s = x_s[:B,:], labels_s[:B,:]
            x_t, labels_t = x_t[:B,:], labels_t[:B,:]

            x_s, x_t, labels_s, labels_t = x_s.cuda(), x_t.cuda(), labels_s.cuda(), labels_t.cuda()
            
            # measure data loading time
            data_time.update(time.time() - end)
            x = torch.cat((x_s, x_t), dim=0)
            assert x.requires_grad is False

            optimizer.zero_grad()    
            
            # compute output
            with autocast():  # mixed precision
                y, f = model(x)
                
            y_s, y_t = y[:B,:], y[B:,:]
            task_loss = asl(y_s, labels_s)    # task loss 
            
            with autocast():  # mixed precision
                discrepancy_loss = discrepancy(f, B) 
                
            transfer_loss = discrepancy_loss * args.trade_off # multiply the lambda to trade off the loss term
            loss = task_loss + transfer_loss

            cls_acc = mAP(labels_s.detach().cpu().numpy(),
                            y_s.detach().cpu().numpy())
            tgt_acc = mAP(labels_t.detach().cpu().numpy(),
                            y_t.detach().cpu().numpy())
            
            ASL.update(task_loss.item(), x_s.size(0)) 
            Disc.update( discrepancy_loss.item(), x_s.size(0))
            cls_accs.update(cls_acc.item(), x_s.size(0))
            tgt_accs.update(tgt_acc.item(), x_t.size(0))

            scaler.scale(loss).backward()
            scaler.step(optimizer)        
            scaler.update()
            scheduler.step()
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)

        model.eval()
        discrepancy.eval()
        
        print('\n Target Validation')
        mAP_score_t = validate(t_val_loader, model, args)       
        
        # saving models 
        torch.save(model.state_dict(), logger.get_checkpoint_path('latest'))
        if mAP_score_t > highest_mAP_t:
            highest_mAP_t = mAP_score_t
            torch.save(model.state_dict(), logger.get_checkpoint_path('best'))
            
        print('\n Source Validation')
        mAP_score_s = validate(s_val_loader, model, args)       
        if mAP_score_s > highest_mAP_s:
            highest_mAP_s = mAP_score_s    
                   
        print('\n------------------------')
        print('Epoch {}/{}'.format(epoch, Epochs))
        print(args.source, ' current_mAP (s) = {:.2f}, highest_mAP (s) = {:.2f}'.format(
            mAP_score_s, highest_mAP_s))
        print(args.target, ' current_mAP (t) = {:.2f}, highest_mAP (t) = {:.2f}'.format(
            mAP_score_t, highest_mAP_t))
        print('------------------------\n')
        
def validate(val_loader, model, args):
    Sig = torch.nn.Sigmoid()
    preds = []
    targets = []
    model.eval()
    for _, (input, target) in enumerate(val_loader):
        if args.source=='coco':
            target = target.max(dim=1)[0]
        # compute output
        with torch.no_grad():
            with autocast():  # mixed precision
                output_regular, _ = model(input.cuda())

        preds.append(Sig(output_regular).detach().cpu())
        targets.append(target.detach().cpu())

    mAP_score = mAP(torch.cat(targets).numpy(),
                            torch.cat(preds).numpy())

    print("mAP score {:.2f}".format(mAP_score))
    
    return mAP_score

def load_model_weights_pretrain(model, model_path):
    state = torch.load(model_path, map_location='cpu')
    print('Loading pre-trained model from ', model_path)
    
    for key in model.state_dict():
        if 'num_batches_tracked' in key:
            continue
        p = model.state_dict()[key]
        if key in state['state_dict']:
            ip = state['state_dict'][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print(
                    'could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
        else:
            print('could not load layer: {}, not in checkpoint'.format(key))
    print('Loaded!\n')
    return model

def load_model_weights_finetune(model, model_path):
    state = torch.load(model_path, map_location='cpu')
    for key in model.state_dict():
        if 'num_batches_tracked' in key:
            continue
        p = model.state_dict()[key]
        if key in state:
            ip = state[key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print(
                    'could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
        else:
            print('could not load layer: {}, not in checkpoint'.format(key))
    return model

def validate_full(val_loader, model, args):
    Sig = torch.nn.Sigmoid()
    tp, fp, fn, tn, count = 0, 0, 0, 0, 0
    preds = []
    targets = []
    model.eval()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        
        # compute output
        with torch.no_grad():
            output, _ = model(input.cuda())
        
        # for mAP calculation
        preds.append(Sig(output.cpu()))
        targets.append(target.cpu())
        output = Sig(output)

        # measure accuracy and record loss
        pred = output.data.gt(args.thre).long()

        tp += (pred + target).eq(2).sum(dim=0)
        fp += (pred - target).eq(1).sum(dim=0)
        fn += (pred - target).eq(-1).sum(dim=0)
        tn += (pred + target).eq(0).sum(dim=0)
        count += input.size(0)

        this_tp = (pred + target).eq(2).sum()
        this_fp = (pred - target).eq(1).sum()
        this_fn = (pred - target).eq(-1).sum()
        this_tn = (pred + target).eq(0).sum()

        this_prec = this_tp.float() / (
            this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
        this_rec = this_tp.float() / (
            this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0

        p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[
                                                                             i] > 0 else 0.0
               for i in range(len(tp))]
        r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[
                                                                             i] > 0 else 0.0
               for i in range(len(tp))]
        f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for
               i in range(len(tp))]

        mean_p_c = sum(p_c) / len(p_c)
        mean_r_c = sum(r_c) / len(r_c)
        mean_f_c = sum(f_c) / len(f_c)

        p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
        r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
        f_o = 2 * p_o * r_o / (p_o + r_o)

    print(
        '--------------------------------------------------------------------')
    print(' * P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
          .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))
    mAP_score = mAP(torch.cat(targets).numpy(), torch.cat(preds).numpy())
    
    print("mAP score:", mAP_score)

if __name__ == '__main__':
    main()
