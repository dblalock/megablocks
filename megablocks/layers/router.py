import collections
import math

from megablocks.layers import common
from megablocks.layers.arguments import Arguments
import torch
import torch.nn.functional as F


# NOTE: To enable end-to-end benchmarking without convergence we
# support a flag to force the router to assign tokens uniformly
# across the experts. We do this with a custom autograd operation
# so that PyTorch still executes the full set of router operation.
class _UniformExpertAssignment(torch.autograd.Function):


    @staticmethod
    def forward(ctx, x, num_experts):
        out = torch.arange(x.numel(), dtype=x.dtype, device=x.device)
        out = torch.remainder(out, num_experts)
        return out.view(x.shape)
_uniform_expert_assignment = _UniformExpertAssignment.apply


class RouterOutput(collections.namedtuple('RouterOutput',
                   'scores expert_weights topk_experts loss'.split())):
    pass


class Router(torch.nn.Module):

    def __init__(self, args : Arguments):
        super().__init__()
        self.args = args

        # Learned router parameters.
        #
        # NOTE: This weight matrix is not parallelized with expert model
        # parallelism. Each device needs the entire router weight matrix
        # so that it can route its batch of data correctly.
        self.layer = torch.nn.Linear(
            args.hidden_size,
            args.moe_num_experts,
            bias=False,
            dtype=common.dtype(args),
            device=args.device)
        args.init_method(self.layer.weight)

    def jitter(self, x):
        low = 1.0 - self.args.moe_jitter_eps
        high = 1.0 + self.args.moe_jitter_eps
        noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
        return low + noise * (high - low)

    def _top_k(self, scores):
        if self.args.moe_top_k == 1:
            return scores.max(dim=-1)
        return torch.topk(scores, self.args.moe_top_k, dim=-1)


class LearnedRouter(Router):

    def forward(self, x) -> RouterOutput:
        if self.training and self.args.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        sl, bs, hs = x.size()
        scores = self.layer(x.view(-1, hs)).softmax(dim=-1)
        expert_weights, expert_indices = self._top_k(scores)

        expert_indices = (
            _uniform_expert_assignment(expert_indices, self.args.moe_num_experts)
            if self.args.uniform_expert_assignment else expert_indices
        )
        return scores, expert_weights, expert_indices, 0


class UnsupervisedRouter(Router):

    def forward(self, x) -> RouterOutput:
        k = self.args.moe_top_k
        num_experts = self.args.moe_num_experts

        x = x.detach()
        if self.training and self.args.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        x_2d = x.reshape(-1, x.shape[-1])
        scores = F.softplus(self.layer(x_2d))  # elemwise >= 0
        scores = F.normalize(scores, p=1, dim=1) # sum to 1
        # scores = F.softmax(self.layer(x_2d), dim=-1)

        expert_weights, expert_indices = self._top_k(scores)
        expert_weights = expert_weights.detach()  # avoid saving scatter input

        ret = RouterOutput(scores, expert_weights, expert_indices, 0)
        if not self.training or num_experts < 2:
            return ret

        # ------------------------ loss

        # spikiness loss; we square the gap between the highest
        # score and 1.0, which makes the gradient proportional to
        # the gap. This is one of only a few simple functions
        # whose gradient *decreases* as we approach full
        # commitment to one expert. This is necessary so that we
        # trade off spikiness and balance losses, instead of
        # creating a positive feedback loop where one dominates.
        uniform_frac = 1. / num_experts
        top_gaps = 1. - expert_weights
        spikiness_loss = (top_gaps * top_gaps).mean()

        # normalize spikiness loss to [0, 1]
        min_spikiness_loss = (1 - 1. / k)**2
        max_spikiness_loss = (1 - uniform_frac)**2
        spikiness_loss = ((spikiness_loss - min_spikiness_loss) /
                          (max_spikiness_loss - min_spikiness_loss))

        # # normalize spikiness loss grad to [0, 1]
        # min_spikiness_loss_grad = 1 - 1. / k
        # max_spikiness_loss_grad = 1 - uniform_frac
        # spikiness_loss = ((spikiness_loss - min_spikiness_loss_grad) /
        #                   (max_spikiness_loss_grad - min_spikiness_loss_grad))

        # balance loss; similarly, we use a function that has
        # a decreasing gradient as we approach perfect balance
        # soft_indicators = torch.softmax(scores, dim=-1)
        soft_indicators = scores
        # score_cutoffs = expert_weights[k - 1].reshape(-1, 1)
        # soft_indicators = scores - score_cutoffs + .001  # >0 iff selected
        # soft_indicators = F.tanh(soft_indicators)
        assignment_fracs = soft_indicators.mean(dim=0)
        frac_diffs = assignment_fracs - uniform_frac
        balance_loss = (frac_diffs * frac_diffs).mean()

        # normalize balance loss to [0, 1]
        # worst-case loss is one expert getting all assignments; best-case is
        # perfect balance, which has 0 loss
        maxval_loss = (1 - uniform_frac) ** 2
        zerovals_loss = (num_experts - 1) * (uniform_frac * uniform_frac)
        max_balance_loss = (maxval_loss + zerovals_loss) / num_experts
        balance_loss /= max_balance_loss

        return ret._replace(loss=(spikiness_loss + balance_loss))
