import torch

def positional_encoding(pos, feat_size=16, timescale=10000):
    '''
    pos : [N X D] matrix of positions

    returns a positional encoding of [N x (D * feat_size)]
    '''

    N, D = pos.shape

    sin_freq = torch.arange(0, feat_size, 2.0) / feat_size
    cos_freq = torch.arange(1, feat_size, 2.0) / feat_size
    sin_freq = 1 / (timescale ** sin_freq)
    cos_freq = 1 / (timescale ** cos_freq)

    sin_emb = torch.sin(torch.einsum('ni,d->ndi', pos, sin_freq))
    cos_emb = torch.cos(torch.einsum('ni,d->ndi', pos, cos_freq))

    encoding = torch.zeros(N, D * feat_size)
    for i in range(D):
        start_idx = i * feat_size
        end_idx   = (i + 1) * feat_size
        encoding[:, start_idx:end_idx:2]   = sin_emb[:, :, i]
        encoding[:, start_idx+1:end_idx:2] = cos_emb[:, :, i]

    return encoding
