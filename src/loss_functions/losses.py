import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Optional, Any, Tuple
from sklearn.mixture import GaussianMixture
import scipy.stats as stats
import numpy as np

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        print('ASL is selected')
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
            
        return -loss.sum()


class GMM_Discrepancy(nn.Module):
    def __init__(self, classifier: nn.Module, args):
        super(GMM_Discrepancy, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.classifier = classifier
        self.reg_0 = args.reg_0
        self.reg_1 = args.reg_1
        
    @staticmethod
    def calculate_frechet_dist(X: torch.Tensor, Y: torch.Tensor):
        var_X, mu_X= torch.var_mean(X)
        var_Y, mu_Y = torch.var_mean(Y)
        return ((mu_X - mu_Y)**2 + (var_X.sqrt() - var_Y.sqrt())**2).float()
    
    def get_pdf(self, data, n_components=2):
        
        # fitting a two component GMM on the predictions
        gm = GaussianMixture(n_components=n_components,reg_covar=0.001).fit(data.reshape(-1,1))

        # extracting two gaussian parameters, one for 0s and other for 1s
        mu_1, mu_2 = gm.means_[np.argmin(gm.means_)][0], gm.means_[np.argmax(gm.means_)][0]
        s_1, s_2 = gm.covariances_[np.argmin(gm.means_)][0][0], gm.covariances_[np.argmax(gm.means_)][0][0]

        # generating two seperate gaussian distriubtions
        x1 = np.linspace(0.0, 0.5, num=300)
        x2 = np.linspace(0.6, 1.0, num=300)
        y_1 = stats.norm.pdf(x1, mu_1, s_1).reshape(-1,1)
        y_2 = stats.norm.pdf(x2, mu_2, s_2).reshape(-1,1)
        
        return  torch.tensor(y_1), torch.tensor(y_2)
        
    def forward(self, f: torch.Tensor, B:int) -> torch.Tensor:
            f_grl = self.grl(f) #reversed features 
            y = self.classifier(f_grl) # classifier will act as a discriminator
            y_s, y_t = y[:B,:], y[B:,:] # seperating source and target predictions

            # Getting two seperate gaussian distributions each for source and target
            y_s1, y_s2 = self.get_pdf(torch.sigmoid(y_s).detach().cpu().numpy())
            y_t1, y_t2 = self.get_pdf(torch.sigmoid(y_t).detach().cpu().numpy())

            # Loss based on Frechet distance
            loss = self.calculate_frechet_dist(y_s1, y_t1)*self.reg_0 + self.calculate_frechet_dist(y_s2, y_t2)*self.reg_1
            return loss
 

class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class WarmStartGradientReverseLayer(nn.Module):
    """Gradient Reverse Layer :math:`\mathcal{R}(x)` with warm start
        The forward and backward behaviours are:
        .. math::
            \mathcal{R}(x) = x,
            \dfrac{ d\mathcal{R}} {dx} = - \lambda I.
        :math:`\lambda` is initiated at :math:`lo` and is gradually changed to :math:`hi` using the following schedule:
        .. math::
            \lambda = \dfrac{2(hi-lo)}{1+\exp(- α \dfrac{i}{N})} - (hi-lo) + lo
        where :math:`i` is the iteration step.
        Args:
            alpha (float, optional): :math:`α`. Default: 1.0
            lo (float, optional): Initial value of :math:`\lambda`. Default: 0.0
            hi (float, optional): Final value of :math:`\lambda`. Default: 1.0
            max_iters (int, optional): :math:`N`. Default: 1000
            auto_step (bool, optional): If True, increase :math:`i` each time `forward` is called.
              Otherwise use function `step` to increase :math:`i`. Default: False
        """

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1
