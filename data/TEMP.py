import numpy as np
import torch


x = torch.zeros([4, 3, 14, 256, 256])
x = x[0, :, [3, 2, 1], ...]
print(x.shape)
y = torch.zeros([4, 10, 256, 256])
y = y[0, [3, 2, 1], ...]
print(y.shape)
out= torch.zeros([4, 1, 10, 256, 256])
out = out[0, 0, [3, 2, 1], ...]
print(out.shape)
in_m = torch.zeros([4, 3, 1, 256, 256])
print('titi', in_m[0, :, ...].shape)
in_m = in_m.squeeze(2)
print(in_m.shape)



"""
x = torch.zeros([4, 1, 13, 256, 256])
x = x.squeeze(1)
x = x[:,1:11,:,:]

t = torch.tensor(3107.6250, device='cuda:0', dtype=torch.float64)
t = t.unsqueeze(-1)
print(t.shape)
t = t.repeat((4,3, 32, 32))
print(t.shape)
bp = t.permute(0,2,3,1).contiguous().view(4 * 32 * 32, 3)
print(bp.shape)
#t = BxTxHxW

sz_b, seq_len, d, h, w = 4, 3, 128, 32, 32
#bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)

#bp = bp.permute(0, 2, 3, 1).contiguous().view(4 * 32 * 32, 3)



t = torch.tensor(3107.6250, device='cuda:0', dtype=torch.float64)
t = t.unsqueeze(-1)
print(t.shape)
t = t.repeat((1, 1, 32))
print(t.shape)


t1 = torch.zeros(4, 3, 10, 3, 3)
t2 = torch.zeros(4, 3, 4, 3, 3)
u = torch.cat([t1, t2], dim=2)
#print(u.shape)

p = torch.tensor([[3088, 3088, 3088],
        [3088, 3136, 3143],
        [3088, 3088, 3143],
        [3088, 3112, 3143],
        [3080, 3085, 3090],
        [3090, 3135, 3145],
        [3085, 3090, 3145],
        [3090, 3110, 3145]], device='cuda:0')

print(p.shape)
p = p.type(torch.float64)
m = p.mean()
print(m.shape, m)
"""