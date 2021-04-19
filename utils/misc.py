import logging
import sys
import os

import torch
import torch.distributed as dist


def setup_process(rank, world_size, master_port='12355'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_process():
    dist.destroy_process_group()


def save_training_data(path, optimizer=None, scaler=None, epoch=None):
    checkpoint = {
        'optimizer': None if optimizer is None else optimizer.state_dict(),
        'scaler': None if scaler is None else scaler.state_dict(),
        'epoch': epoch
    }

    torch.save(checkpoint, os.path.join(path, 'training_data.pt'))


def load_training_data(path, optimizer=None, scaler=None, map_location=None):
    checkpoint = torch.load(os.path.join(path, 'training_data.pt'), map_location=map_location)

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if scaler is not None and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])

    return checkpoint


def setup_logger(name, log_file, level=loggin.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
        
    return logger
