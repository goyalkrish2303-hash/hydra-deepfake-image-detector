import torch
import torch.nn.functional as F


def gradcam(model, x):
    """Grad-CAM for the fake logit. x: (1,3,224,224). Returns heatmap and fake score."""
    model.eval()
    with torch.enable_grad():
        x = x.clone().requires_grad_(True)
        logit, fmap = model(x, return_fmap=True)
        grads = torch.autograd.grad(logit.sum(), fmap)[0]
    w = grads.mean((2, 3), keepdim=True)
    cam = F.relu((w * fmap).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy(), torch.sigmoid(logit).item()
