"""
SUMMARY:  mu-law transform
AUTHOR:   Qiuqiang Kong
Created:  2016.10.02
Modified: 2017.03.30 add annotations
          2017.04.22 modify methods
          2018.03.10 Rewrite
--------------------------------------
"""
import numpy as np
        
        
class MuLaw(object):
    def __init__(self, mu):
        self.mu = mu
        
    def transform(self, x):
        mu = self.mu
        return np.sign(x) * (np.log(1. + mu * np.abs(x)) / np.log(1. + mu))
        
    def inverse_transform(self, x):
        mu = self.mu
        sign = (np.sign(x) + 1) / 2.
        return sign * (np.power(1 + mu, x) - 1.) / mu + \
               (1. - sign) * (1. - np.power(1. + mu, -x)) / mu
     
        
class Quantize(object):
    def __init__(self, quantize):
        self.quantize = quantize
        self.scale = 1.
        self.interval = 2. * self.scale / self.quantize
        
    def transform(self, x):
        assert(np.max(np.abs(x)) <= self.scale)
        return np.floor((x - (-self.scale)) / self.interval).astype(np.int32)
        
    def inverse_transform(self, x):
        assert(np.min(x) >= 0)
        assert(np.max(x) < self.quantize)
        return x * self.interval + (-self.scale)
    
        
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    
    # ITU-T G.711. 
    if False:
        y = np.array([0, 1, 30, 64, 222, 478, 990, 2014, 4062, 8158])    
        z = np.array([0, 1, 16, 32, 48, 64, 80, 96, 112, 128])
        plt.plot(y, z)
        plt.title("ITU-T G.711.")
        plt.show()
    
    # Mu-law transform
    if False:
        mulaw = MuLaw(mu=256)
        
        y = np.arange(0, 1, 0.01)
        mu_y = mulaw.transform(y)
        inv_mu_y = mulaw.inverse_transform(mu_y)
        
        fig, axs = plt.subplots(2,1, sharex=True)
        axs[0].plot(y, mu_y)
        axs[1].plot(y, inv_mu_y)
        plt.show()
    
    # Quantize
    if True:
        qu = Quantize(quantize=256)
        y = np.arange(-1., 1., 0.005)
        digit_y = qu.transform(y)
        inverse_digit_y = qu.inverse_transform(digit_y)
        
        print(digit_y)
        plt.plot(y, inverse_digit_y)
        plt.show()
    