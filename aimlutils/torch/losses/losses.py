import torch, logging
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)

###################
#
# Entropy Losses
#
###################

def loss_fn(recon_x, x, mu, logvar):
    criterion = nn.BCELoss(reduction='sum')    
    BCE = criterion(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD


class SymmetricCE:
    
    def __init__(self, alpha, gamma, kld_weight = 1.0):
        self.alpha = alpha
        self.gamma = alpha
        self.kld_weight = kld_weight
        
        logger.info(f"Loaded Symmetric Cross Entropy loss ...")
        logger.info(f"... with alpha = {alpha}, gamma = {gamma}, and kld_weight = {kld_weight}")

    def __call__(self, recon_x, x, mu, logvar):
        criterion = nn.BCELoss(reduction='sum')    
        BCE = criterion(recon_x, x)
        #KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return self.alpha * BCE + self.kld_weight * self.gamma * KLD, BCE, KLD
    
class SymmetricMSE:
    
    def __init__(self, alpha, gamma, kld_weight = 1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.kld_weight = kld_weight
        
        logger.info(f"Loaded Symmetric MSE loss ...")
        logger.info(f"... with alpha = {alpha}, gamma = {gamma}, and kld_weight = {kld_weight}")

    def __call__(self, recon_x, x, mu, logvar):
        criterion = nn.MSELoss(reduction='sum')  
        BCE = criterion(recon_x, x)
        #KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return self.alpha * BCE + self.kld_weight * self.gamma * KLD, BCE, KLD
    

###################
#
# Regression Losses - https://github.com/tuantle/regression-losses-pytorch
#
###################

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


class XTanhLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(ey_t * torch.tanh(ey_t))


class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t)