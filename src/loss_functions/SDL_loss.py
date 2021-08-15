import torch
import torch.nn as nn


class SDLLoss(nn.Module):
    def __init__(self, weight=0.3, reduction='mean', wordvec_array=None):
        super(SDLLoss, self).__init__()
        self.eps = 1e-7
        assert reduction in ['mean', 'sum']
        self.reduction_type = reduction
        self.reduction = torch.mean if reduction == 'mean' else torch.sum
        self.wordvec_array = wordvec_array
        self.embed_len = 300
        self.weight = weight

    def forward(self, x, y):
        # Update Aux Loss weight
        # if not provided the aux loss will not be used (regularization of matrix var)
        use_var_aux_loss = False
        if self.weight is not None:
            use_var_aux_loss = True
            weight = self.weight

        # Compute dot product for the output matrix A with every wordvec in the dictionary
        # init loss variable
        batch_size = y.size()[0]
        l = torch.zeros(batch_size).cuda()

        # X is in a vector of size kx300, we need to unflat it into a matrix A
        un_flat = x.view(x.shape[0], self.embed_len, -1)

        k = un_flat.shape[2]  # k = number of rows in the matrix

        # Compute dot product between A and all available word vectors (number of tags x 300)
        dot_prod_all = [torch.sum((un_flat[:, :, i].unsqueeze(2) * self.wordvec_array), dim=1).unsqueeze(2) for i in
                        range(k)]

        # Apply max on A dot wordvecs
        dot_prod_all = torch.max(torch.cat(dot_prod_all, dim=2), dim=-1)
        dot_prod_all = dot_prod_all.values

        # For loop over all batch
        for i in range(0, batch_size):
            # Separate Positive and Negative labels
            dot_prod_pos = dot_prod_all[i, y[i] == 1]  # y==1 means positive labels
            dot_prod_neg = dot_prod_all[i, (1 - y[i]).bool()]  # unknown are treated as negatives (-1,0)
            # dot_prod_neg = dot_prod_all[i, y[i] == 0]  # unknown are not used as negatives (0)

            # Compute v = max(An) - max(Ap)
            # v.shape = [num_pos, num_negatives]
            if len(dot_prod_neg) == 0:  # if no negative labels
                v = -dot_prod_pos.unsqueeze(1)
            else:
                v = dot_prod_neg.unsqueeze(0) - dot_prod_pos.unsqueeze(1)

            # Final loss equation (1/num_classes) * sum(log(1+exp(max(An_i) - max(Ap_i))))
            num_pos = dot_prod_pos.shape[0]
            # num_neg = dot_prod_neg.shape[0]
            total_var = calc_diversity(self.wordvec_array, y[i])

            l[i] = (1 + total_var) * torch.sum(torch.log(1 + torch.exp(v))) / (num_pos)

            if use_var_aux_loss:  # compute variance based auxiliary loss
                l1_err = var_regularization(un_flat[i])
                l[i] = 2 * ((1 - weight) * (l[i]) + weight * l1_err)

        return self.reduction(l)


def calc_diversity(wordvec_array, y_i):
    rel_vecs = wordvec_array[:, :, y_i == 1]
    rel_vecs = rel_vecs.squeeze(0)
    if rel_vecs.shape[1] == 1:
        sig = rel_vecs * 0  # det_c = 0
    else:
        sig = torch.var(rel_vecs, dim=1)

    return sig.sum()


def var_regularization(x_i):
    sig2 = torch.var(x_i, dim=1)
    l1_err = torch.norm(sig2, dim=-1, p=1)

    return l1_err