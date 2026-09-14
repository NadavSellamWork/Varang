import torch

class ParametersDistribution:
    def __init__(self, order):
        self.order = order

    def sample(self, num_sets:int):
        """
        this function will return a ${num_sets} sets of parameters according to the distribution
        """
        pass

class AR1UniformParameterDistribution(ParametersDistribution):
    def __init__(self, min_coeff=0.1, max_coeff=0.9):
        super().__init__(1)
        self.min_coeff = min_coeff
        self.max_coeff = max_coeff
    
    def sample(self, num_sets: int):
        return self.min_coeff + torch.rand(num_sets) * (self.max_coeff - self.min_coeff)

DEFAULT_CENTERS = [
    torch.tensor([0.1, 0.2], dtype=torch.float64),
    torch.tensor([0.3, 0.4], dtype=torch.float64),
]

DEFAULT_COVS = [
    torch.tensor([[0.004, 0.002],
                  [0.002, 0.008]], dtype=torch.float64),
    torch.tensor([[0.015, -0.01],
                  [-0.01, 0.012]], dtype=torch.float64),
]

DEFAULT_WEIGHTS = torch.tensor([0.3, 0.7], dtype=torch.float64)

class AR2CorrelatedParameterDistribution(ParametersDistribution):
    def __init__(
        self,
        centers=DEFAULT_CENTERS,
        covariances=DEFAULT_COVS,
        weights=DEFAULT_WEIGHTS,
        max_coeffs_sum=0.93,
        oversample_factor=2.0,
    ):
        super().__init__(2)
        self.centers = [torch.as_tensor(c, dtype=torch.float32) for c in centers]
        self.covariances = [torch.as_tensor(c, dtype=torch.float32) for c in covariances]
        self.weights = torch.as_tensor(weights, dtype=torch.float32)
        self.max_coeffs_sum = float(max_coeffs_sum)
        self.oversample_factor = float(oversample_factor)

        w = self.weights / self.weights.sum()
        self.cdf = torch.cumsum(w, dim=0)

        self.D = int(self.centers[0].numel())
        assert all(int(c.numel()) == self.D for c in self.centers)
        assert len(self.centers) == len(self.covariances) == int(self.weights.numel())

    @torch.no_grad()
    def sample(self, num_sets: int, generator=None) -> torch.Tensor:
        num_sets = int(num_sets)
        if num_sets <= 0:
            return torch.empty((0, self.D), dtype=torch.float32)

        K = int(self.weights.numel())

        # choose component per sample via inverse CDF
        u = torch.rand(num_sets, generator=generator)
        comp = torch.searchsorted(self.cdf, u).clamp_max(K - 1)  # [num_sets]

        out = torch.empty((num_sets, self.D), dtype=torch.float32)

        # sample each chosen component, with rejection for stability constraint
        for k in range(K):
            idx = (comp == k).nonzero(as_tuple=False).squeeze(-1)
            m = int(idx.numel())
            if m == 0:
                continue

            dist = torch.distributions.MultivariateNormal(
                self.centers[k],
                covariance_matrix=self.covariances[k],
            )

            got = []
            total = 0
            batch = max(512, int(m * self.oversample_factor))

            while total < m:
                samp = dist.sample((batch,), generator=generator) if generator is not None else dist.sample((batch,))
                samp = samp[samp.abs().sum(dim=1) <= self.max_coeffs_sum]
                if samp.numel() == 0:
                    continue
                got.append(samp)
                total += int(samp.shape[0])

            out[idx] = torch.cat(got, dim=0)[:m]

        # shuffle rows so even within a batch you don't get block structure
        perm = torch.randperm(num_sets, generator=generator)
        return out[perm]