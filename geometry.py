import torch


def circle_iou_concentric(r1, r2, eps=1e-12):
    #print(r1, r2)
    r1 = r1.abs(); r2 = r2.abs()
    mn = torch.minimum(r1, r2).pow(2)
    mx = torch.maximum(r1, r2).pow(2)
    return mn / mx.clamp_min(eps)

def project_from_inside_cube(a, b, eps=1e-12):
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=a.dtype)
    d = b - a

    inv_d = torch.where(d.abs() > eps, 1.0 / d, torch.full_like(d, float("inf")))
    t1 = (-1.0 - a) * inv_d
    t2 = ( 1.0 - a) * inv_d
    tmax_axis = torch.maximum(t1, t2)
    t_exit = tmax_axis.amin(dim=-1)
    p = a + t_exit.unsqueeze(-1) * d

    inside_b = (b.abs() <= 1.0).all(dim=-1)
    p = torch.where(inside_b.unsqueeze(-1), b, p.detach())
    return p

def project_from_inside_sphere(a, b, eps=1e-12):
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=a.dtype)
    d = b - a

    ad = (a * d).sum(dim=-1)
    dd = (d * d).sum(dim=-1)
    aa = (a * a).sum(dim=-1)

    disc = ad**2 - dd * (aa - 1.0)
    disc = torch.clamp(disc, min=0)
    t_exit = (-ad + torch.sqrt(disc)) / (dd + eps)

    p = a + t_exit.unsqueeze(-1) * d

    inside_b = (b.norm(dim=-1) <= 1.0 + eps)
    p = torch.where(inside_b.unsqueeze(-1), b, p.detach())
    return p