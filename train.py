#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import platform
import torch
import torch.nn as nn
import os
import time
import sys
from loguru import logger
import argparse
from tqdm import tqdm
from pathlib import Path
from data_handling.audiocaps_dataset import get_audiocaps_loader
from tools.config_loader import get_config
from tools.utils import *
from tools.file_io import load_pickle_file
from pprint import PrettyPrinter
from warmup_scheduler import GradualWarmupScheduler
from eval_metrics import evaluate_metrics
from models.TransModel import ACT
from tools.beam import beam_decode


def train():

    start_time = time.time()

    batch_losses = AverageMeter()

    model.train()

    for batch_idx, train_batch in tqdm(enumerate(training_data), total=len(training_data)):

        src, tgt, tgt_len, f_names, captions = train_batch
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_pad_mask = set_tgt_padding_mask(tgt, tgt_len)

        optimizer.zero_grad()

        y_hat = model(src, tgt, target_padding_mask=tgt_pad_mask)
        tgt = tgt[:, 1:]  # exclude <sos>
        y_hat = y_hat.transpose(0, 1)  # batch x words_len x ntokens
        y_hat = y_hat[:, :tgt.size()[1], :]  # truncate to the same length with target

        loss = criterion(y_hat.contiguous().view(-1, y_hat.size()[-1]),
                         tgt.contiguous().view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_grad)
        optimizer.step()
        batch_losses.update(loss.cpu().item())

    elapsed_time = time.time() - start_time
    epoch_loss = batch_losses.avg
    current_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]

    main_logger.info('epoch: {}, train_loss: {:.4f}, time elapsed: {:.4f}, lr:{:02.2e}'.
                     format(epoch, epoch_loss, elapsed_time, current_lr))


def eval_greedy(data):

    model.eval()
    with torch.no_grad():
        start_time = time.time()
        y_hat_all = []
        ref_captions_dict = []
        file_names_all = []

        for batch_idx, eval_batch in tqdm(enumerate(data), total=len(data)):

            src, target_dicts, file_names = eval_batch
            src = src.to(device)
            output = greedy_decode(model, src, sos_ind=sos_ind, eos_ind=eos_ind)

            output = output[:, 1:].int()
            y_hat_batch = torch.zeros(output.shape).fill_(eos_ind).to(device)

            for i in range(output.shape[0]):    # batch_size
                for j in range(output.shape[1]):
                    y_hat_batch[i, j] = output[i, j]
                    if output[i, j] == eos_ind:
                        break
                    elif j == output.shape[1] - 1:
                        y_hat_batch[i, j] = eos_ind

            y_hat_batch = y_hat_batch.int()
            y_hat_all.extend(y_hat_batch.cpu())
            ref_captions_dict.extend(target_dicts)
            file_names_all.extend(file_names)

        eval_time = time.time() - start_time
        captions_pred, captions_gt = decode_output(y_hat_all, ref_captions_dict,
                                                   file_names_all, words_list)
        greedy_metrics = evaluate_metrics(captions_pred, captions_gt)
        spider = greedy_metrics['spider']['score']
        cider = greedy_metrics['cider']['score']
        main_logger.info(f'cider: {cider:7.4f}')
        main_logger.info(f'Spider score using greedy search: {spider:7.4f}, eval time: {eval_time:.4f}')


def eval_beam(data, beam_size):

    model.eval()
    with torch.no_grad():
        start_time = time.time()
        y_hat_all = []
        ref_captions_dict = []
        file_names_all = []

        for batch_idx, eval_batch in tqdm(enumerate(data), total=len(data)):

            src, target_dicts, file_names = eval_batch
            src = src.to(device)
            output = beam_decode(src, model, sos_ind, eos_ind, beam_width=beam_size)

            output = output[:, 1:].int()
            y_hat_batch = torch.zeros(output.shape).fill_(eos_ind).to(device)

            for i in range(output.shape[0]):  # batch_size
                for j in range(output.shape[1]):
                    y_hat_batch[i, j] = output[i, j]
                    if output[i, j] == eos_ind:
                        break
                    elif j == output.shape[1] - 1:
                        y_hat_batch[i, j] = eos_ind

            y_hat_batch = y_hat_batch.int()
            y_hat_all.extend(y_hat_batch.cpu())
            ref_captions_dict.extend(target_dicts)
            file_names_all.extend(file_names)

        eval_time = time.time() - start_time
        captions_pred, captions_gt = decode_output(y_hat_all, ref_captions_dict, file_names_all, words_list, beam=True)
        beam_metrics = evaluate_metrics(captions_pred, captions_gt)
        spider = beam_metrics['spider']['score']
        cider = beam_metrics['cider']['score']
        main_logger.info(f'cider: {cider:7.4f}')
        main_logger.info(f'Spider score using beam search (beam size:{beam_size}): {spider:7.4f}, eval time: {eval_time:.4f}')
        spiders.append(spider)
        if config.mode != 'eval':
            if beam_size == 3 and (epoch % 5) == 0:
                for metric, values in beam_metrics.items():
                    main_logger.info(f'beam search (size 3): {metric:<7s}: {values["score"]:7.4f}')
            if spider >= max(spiders):
                torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "beam_size": beam_size,
                        "epoch": epoch,
                        }, str(model_output_dir) + '/best_model.pth'.format(epoch))
        else:
            if spider >= max(spiders):
                eval_metrics['metrics'] = beam_metrics
                eval_metrics['beam_size'] = beam_size


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser(description='Settings for ACT training')

parser.add_argument('-n', '--exp_name', type=str, default='exp1', help='name of the experiment')

args = parser.parse_args()

config = get_config()

setup_seed(config.training.seed)

exp_name = args.exp_name

# output setting
model_output_dir = Path('outputs', exp_name, 'model')
log_output_dir = Path('outputs', exp_name, 'logging')

model_output_dir.mkdir(parents=True, exist_ok=True)
log_output_dir.mkdir(parents=True, exist_ok=True)

logger.remove()

logger.add(sys.stdout, format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
           filter=lambda record: record['extra']['indent'] == 1)

logger.add(log_output_dir.joinpath('output.txt'), format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
           filter=lambda record: record['extra']['indent'] == 1)

logger.add(str(log_output_dir) + '/captions.txt', format='{message}', level='INFO',
           filter=lambda record: record['extra']['indent'] == 2,
           rotation=rotation_logger)

logger.add(str(log_output_dir) + '/beam_captions.txt', format='{message}', level='INFO',
           filter=lambda record: record['extra']['indent'] == 3,
           rotation=rotation_logger)

main_logger = logger.bind(indent=1)

printer = PrettyPrinter()

device, device_name = (torch.device('cuda'),
                       torch.cuda.get_device_name(torch.cuda.current_device())) \
    if torch.cuda.is_available() else ('cpu', platform.processor())

main_logger.info(f'Process on {device_name}')

words_list = load_pickle_file(config.path.vocabulary)

training_data = get_audiocaps_loader('train', config)
validation_data = get_audiocaps_loader('val', config)
test_data = get_audiocaps_loader('test', config)

ntokens = len(words_list)
sos_ind = words_list.index('<sos>')
eos_ind = words_list.index('<eos>')

main_logger.info('Training setting:\n'
                 f'{printer.pformat(config)}')

model = ACT(config, ntokens)
model.to(device)

main_logger.info(f'Model:\n{model}\n')
main_logger.info('Total number of parameters:'
                 f'{sum([i.numel() for i in model.parameters()])}')

main_logger.info(f'Len of training data: {len(training_data)}')
main_logger.info(f'Len of validation data: {len(validation_data)}')
main_logger.info(f'Len of test data: {len(test_data)}')

if config.training.label_smoothing:
    criterion = LabelSmoothingLoss(ntokens, smoothing=0.1)
else:
    criterion = nn.CrossEntropyLoss()

spiders = []

if config.mode == 'train':

    main_logger.info('Training mode.')
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=config.training.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)

    epochs = config.training.epochs
    ep = 1
    # warm up issue
    optimizer.zero_grad()
    optimizer.step()

    for epoch in range(ep, epochs + 1):

        scheduler_warmup.step(epoch)

        main_logger.info(f'Training epoch {epoch}...')
        train()
        main_logger.info('Metrics on validation set')
        eval_greedy(validation_data)
        eval_beam(validation_data, beam_size=2)
        eval_beam(validation_data, beam_size=3)
    main_logger.info('Training done.')
    best_checkpoint = torch.load(str(model_output_dir) + '/best_model.pth')
    model.load_state_dict(best_checkpoint['model'])
    best_epoch = best_checkpoint['epoch']
    main_logger.info(f'Best checkpoint in {best_epoch} th epoch.')
    main_logger.info('Metrics on test set')
    eval_greedy(test_data)
    eval_beam(test_data, beam_size=2)
    eval_beam(test_data, beam_size=3)
    main_logger.info('Evaluation done.')
elif config.mode == 'eval':
    eval_metrics = {}
    main_logger.info('Evaluation mode')
    model.load_state_dict(torch.load(config.path.eval_model)['model'])
    main_logger.info(f'Weights loaded from {config.path.eval_model}')
    eval_greedy(test_data)
    eval_beam(test_data, beam_size=2)
    eval_beam(test_data, beam_size=3)
    main_logger.info(f"Best metrics with beam size {eval_metrics['beam_size']}:")
    for metric, values in eval_metrics['metrics'].items():
        main_logger.info(f'{metric:<7s}: {values["score"]:7.4f}')
