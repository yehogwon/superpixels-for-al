import torch

def _sample_uncertainty(preds: torch.Tensor) -> torch.Tensor:
    # preds: (N, C, H, W)
    # return: (N)

    top2_probs = torch.topk(preds, k=2, dim=1)[0] # (N, 2, H, W)
    bests = top2_probs[:, 0, :, :] # (N, H, W)
    second_bests = top2_probs[:, 1, :, :] # (N, H, W)
    best_versus_second_best = bests / second_bests # (N, H, W)
    uncertainty = torch.mean(best_versus_second_best, dim=(1, 2)) # (N)
    return uncertainty

def sample_uncertainty(preds: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
    # preds: (N, C, H, W)
    # mask: (N, H, W)
    # return: (N)

    top2_probs = torch.topk(preds, k=2, dim=1)[0] # (N, 2, H, W)
    bests = top2_probs[:, 0, :, :] # (N, H, W)
    second_bests = top2_probs[:, 1, :, :] # (N, H, W)
    best_versus_second_best = bests / second_bests # (N, H, W)
    if mask is not None:
        best_versus_second_best = best_versus_second_best * mask
    uncertainty = torch.mean(best_versus_second_best, dim=(1, 2)) # (N)
    return uncertainty
