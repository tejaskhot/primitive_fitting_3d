import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment


class HungarianLoss(nn.Module):

    def __init__(self):
        super(HungarianLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts, counts):
        batch_size = preds.shape[0]
        # P : (batch_size, gts.shape[1], preds.shape[1])
        loss = []
        for b in range(batch_size):
            count = counts[b]
            P = self.batch_pairwise_dist(
                gts[b, :count, :].unsqueeze(0), preds[b, :count, :].unsqueeze(0)
            )
            cost = P[0].data.cpu().numpy()
            # get optimal assignments by Hungarian algo
            row_ind, col_ind = linear_sum_assignment(cost)
            # get the assignment indices but retrieve values from
            # the GPU tensor; this keeps things differentiable
            for i in range(len(row_ind)):
                loss.append(P[0, row_ind[i], col_ind[i]])
        return torch.stack(loss).mean()

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)

        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = rx.transpose(2, 1) + ry - 2 * zz
        return P


if __name__ == "__main__":

    p1 = Variable(torch.rand(32, 6, 7).uniform_(-0.01, 0.01).cuda(), requires_grad=True)
    p2 = Variable(torch.rand(32, 5, 7).uniform_(0.3, 0.7).cuda())

    hungarian_loss = HungarianLoss()
    counts = (torch.ones(32) * 4).type(torch.LongTensor)
    counts[:5] = 6
    counts[12:14] = 2
    counts = Variable(counts.cuda())
    d = hungarian_loss(p1, p2, counts)
    print(d)
    d.backward()
