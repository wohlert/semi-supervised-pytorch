import torch
import torch.nn as nn
import torch.nn.functional as F


class PlanarNormalizingFlow(nn.Module):
    """
    Planar normalizing flow [Rezende & Mohamed 2015].
    Provides a tighter bound on the ELBO by giving more expressive
    power to the approximate distribution, such as by introducing
    covariance between terms.
    """
    def __init__(self, in_features):
        super(PlanarNormalizingFlow, self).__init__()
        self.u = nn.Parameter(torch.randn(in_features))
        self.w = nn.Parameter(torch.randn(in_features))
        self.b = nn.Parameter(torch.ones(1))

    def forward(self, z):
        # Create uhat such that it is parallel to w
        uw = torch.dot(self.u, self.w)
        muw = -1 + F.softplus(uw)
        uhat = self.u + (muw - uw) * torch.transpose(self.w, 0, -1) / torch.sum(self.w ** 2)

        # Equation 21 - Transform z
        zwb = torch.mv(z, self.w) + self.b

        f_z = z + (uhat.view(1, -1) * F.tanh(zwb).view(-1, 1))

        # Compute the Jacobian using the fact that
        # tanh(x) dx = 1 - tanh(x)**2
        psi = (1 - F.tanh(zwb)**2).view(-1, 1) * self.w.view(1, -1)
        psi_u = torch.mv(psi, uhat)

        # Return the transformed output along
        # with log determninant of J
        logdet_jacobian = torch.log(torch.abs(1 + psi_u) + 1e-8)

        return f_z, logdet_jacobian


class NormalizingFlows(nn.Module):
    """
    Presents a sequence of normalizing flows as a torch.nn.Module.
    """
    def __init__(self, in_features, flow_type=PlanarNormalizingFlow, n_flows=1):
        super(NormalizingFlows, self).__init__()
        self.flows = nn.ModuleList([flow_type(in_features) for _ in range(n_flows)])

    def forward(self, z):
        log_det_jacobian = []

        for flow in self.flows:
            z, j = flow(z)
            log_det_jacobian.append(j)

        return z, sum(log_det_jacobian)