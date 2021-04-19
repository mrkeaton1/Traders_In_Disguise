from datetime import timedelta, datetime
import numpy as np
import math
import torch


def daterange(start_date: datetime, end_date: datetime):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range((d_model+1) // 2):
            pe[pos, 2*i] = math.sin(pos / 10000 ** (2*i/d_model))
            if 2*i+1 < d_model:
                pe[pos, 2*i+1] = math.cos(pos / 10000 ** (2*i/d_model))
    return pe


def recover_true_values(values, vmin, vmax):
    vmin = torch.tensor(vmin).to('cuda')
    vmax = torch.tensor(vmax).to('cuda')
    return values * (vmax - vmin)[None, None, :] + vmin[None, None, :]


def elapsed_time(seconds, short=False):
    if not short:
        minutes = int(seconds // 60)
        hours = int(minutes // 60)
        e_time = ''
        if hours > 0:
            if hours == 1:
                e_time += '1 hour, '
            else:
                e_time += '{:d} hours, '.format(hours)
            minutes %= 60
            seconds %= 60
        if minutes > 0:
            if minutes == 1:
                e_time += '1 minute, '
            else:
                e_time += '{:d} minutes, '.format(minutes)
            seconds %= 60
        if seconds == 1:
            e_time += '1 second.'
        else:
            e_time += '{:.1f} seconds.'.format(seconds)
        return e_time
    else:
        minutes = int(seconds // 60)
        hours = int(minutes // 60)
        e_time = ''
        if hours > 0:
            if hours == 1:
                e_time += '1hr '
            else:
                e_time += '{:d}hrs '.format(hours)
            minutes %= 60
            seconds %= 60
        if minutes > 0:
            if minutes == 1:
                e_time += '1m '
            else:
                e_time += '{:d}m '.format(minutes)
            seconds %= 60
        if seconds == 1:
            e_time += '1s'
        else:
            e_time += '{:.1f}s'.format(seconds)
        return e_time
