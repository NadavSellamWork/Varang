import torch

"""
    these function allow for taking the Fourier transform and inverse of a real signal and keeping the representation as real numbers with the same 
    tensor shape as the input
"""
def fourier_transform(x):
        fft_curve_slices = torch.fft.rfft(x, dim=-1)
        fft_curve_slices = torch.cat([fft_curve_slices.real, fft_curve_slices.imag[...,1:-1]], dim=-1)
        return fft_curve_slices

def inverse_fourier_transform(x):
    original_shape = (x.shape[-1] + 2) // 2
    real, imag = x[...,:original_shape], x[...,original_shape:]
    imag = torch.nn.functional.pad(imag, [1,1,0,0,0,0])
    freq = real + imag * 1j
    return torch.fft.irfft(freq)
