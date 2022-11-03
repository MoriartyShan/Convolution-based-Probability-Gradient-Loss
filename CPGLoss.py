import torch
import torch.nn as nn
import torch.nn.functional as F


_ksize = 3
sobelKernel3 = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)/2.0
sobelKernel5 = torch.tensor(
  [-5,  -4,  0,   4,   5,
  -8, -10,  0,  10,   8,
  -10, -20,  0,  20,  10,
  -8, -10,  0,  10,   8,
  -5,  -4,  0,   4,   5], dtype=torch.float32).view(5,5)/20.0
sobelKernel7 = torch.tensor(
  [-3/18, -2/13, -1/10, 0,  1/10, 2/13, 3/18,
  -3/13, -2/8,  -1/5,  0,  1/5,  2/8,  3/13,
  -3/10, -2/5,  -1/2,  0,  1/2,  2/5,  3/10,
  -3/9,  -2/4,  -1/1,  0,  1/1,  2/4,  3/9,
  -3/10, -2/5,  -1/2,  0,  1/2,  2/5,  3/10,
  -3/13, -2/8,  -1/5,  0,  1/5,  2/8,  3/13,
  -3/18, -2/13, -1/10, 0,  1/10, 2/13, 3/18], dtype=torch.float32).view(7,7)

sobelKernels = [sobelKernel3, sobelKernel5, sobelKernel7]

class SobelLabel(nn.Module):
  def __init__(self, ksize=_ksize):
    super(SobelLabel, self).__init__()
    self.ksize = ksize

    conv = nn.Conv2d(
      in_channels=1,
      out_channels=2,
      kernel_size=(self.ksize, self.ksize),
      groups=1,
      bias=False,
      padding=(self.ksize//2, self.ksize//2),
      padding_mode='replicate')
    conv.requires_grad_(False)

    if (self.ksize == 3):
      sobelKernel = sobelKernels[0]
    elif (self.ksize == 5):
      sobelKernel = sobelKernels[1]
    elif (self.ksize == 7):
      sobelKernel = sobelKernels[2]

    conv.weight[0, 0] = sobelKernel
    conv.weight[1, 0] = sobelKernel.transpose(0, 1)
    self.conv = conv

  def forward(self, one_hot_label:torch.Tensor):
    '''
    @one_hot_label[batch, num_classes, height, width]
    @return: [batch, num_classes, 2, height, width]
    '''
    shape = one_hot_label.shape
    one_hot_label = one_hot_label.contiguous().\
      view(shape[0] * shape[1], 1, shape[2], shape[3])
      # [batch*num_classes, 1, height, width]
    sobel = self.conv(one_hot_label)  # [batch*num_classes, 2, height, width]

    return sobel.view(shape[0],shape[1], 2, shape[2], shape[3])


class CPGLoss(nn.Module):
  def __init__(self, ksize = _ksize):
    super(CPGLoss, self).__init__()
    self.sobel = SobelLabel(ksize)

  def forward(
      self,
      pred: torch.Tensor,
      label: torch.Tensor,
      ignore_index=-1
  ):
    '''
    @pred: [b, c, h, w]
    @label: [b, h, w]
    @ignore_index: ignore the loss of pixels, whose index is ignore_index
    usage:
        >>> criteria = CPGLoss()
        >>> logits = torch.randn(2, 19, 512, 512) # nchw, float
        >>> lbs = torch.randint(0, 19, (2, 512, 512)) # nhw, int64_t
        >>> loss = criteria(logits, lbs)
    '''
    # return torch.zeros(1).cuda()
    num_classes = pred.shape[1]
    valid = (label != ignore_index) #[b, h, w]
    gt = F.one_hot(label.long() * valid,
                num_classes=num_classes).float()  # B, H, W, self.num_classes
    valid = valid.unsqueeze(dim=1) #[b, 1, h, w]
    gt = gt.permute(0, 3, 1, 2) * valid #B, self.num_classes, H, W

    valid = valid.unsqueeze(dim=1)  # b, 1, 1, h, w
    pred_4D_softmax = F.softmax(pred, dim=1)  # b, c, h, w

    pred_grad = self.sobel(pred_4D_softmax)  # b, c, 2, h, w
    gt_grad = self.sobel(gt).detach()  # b, c, 2, h, w

    edge_gt_valid = ((gt_grad * valid) != 0)  # b, c, 2, h, w

    mse = F.mse_loss(gt_grad, pred_grad, reduction='none')[
      edge_gt_valid].mean()

    return mse
if __name__ == "__main__":
  criteria = CPGLoss()
  logits = torch.randn(2, 19, 512, 512)  # nchw, float
  lbs = torch.randint(0, 19, (2, 512, 512))  # nhw, int64_t
  loss = criteria(logits, lbs)
  print("loss,", loss)