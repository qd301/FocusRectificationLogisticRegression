# =================================================================================
# Copyright 2019
# Single label multi-class image classification by deep logistic regression Authors.
# All Rights Reserved.

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =================================================================================
from __future__ import absolute_import
import torch
from torch import nn



class HS_LR(nn.Module):
    """Implementation of Focus rectified logistic regression loss with Hard Selection(HS-LR)
        Args:
        alpha: cost-sensitive hyperparameter in Eq. (5);
        topratio: m % in Eq. (5) for hard selection of most confusing negative classes.
    """
    def __init__(self, num_classes, alpha, topratio, use_gpu=True):
        super(HS_LR, self).__init__()
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.sigmoid = nn.Sigmoid()
        self.alpha = alpha
        self.topratio = topratio

    def forward(self, inputs, targets):
        """
        Args:
            inputs: outputs of the last FC layer, with shape [batch_size, num_classes] (without normalisation)
            targets: ground truth labels with shape (num_classes)
        """
        eps = 1e-7
        probs = self.sigmoid(inputs)
        # Note that for the multi-label case, there is no need to convert scalar label to one-hot label.
        # Comment this line.
        targets = torch.zeros(probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)#one hot embedding

        if self.use_gpu: targets = targets.cuda()
        pred_p = torch.log(probs+eps).cuda()
        pred_n = torch.log(1.0-probs+eps).cuda()


        topk = int(self.num_classes * self.topratio)
        targets = targets.cuda()
        count_pos = targets.sum().cuda()
        hard_neg_loss = -1.0 * (1.0-targets) * pred_n
        topk_neg_loss = -1.0 * hard_neg_loss.topk(topk, dim=1)[0]#topk_neg_loss with shape batchsize*topk

        loss = (targets * pred_p).sum() / count_pos + self.alpha*(topk_neg_loss.cuda()).mean()

        return -1.0*loss



class SS_LR(nn.Module):
    """Implementation of Focus rectified logistic regression loss with Soft Selection(SS-LR)
        Args:
        alpha: cost-sensitive hyperparameter in Eq. (6);
        gamma: gamma in Eq. (6), the temperature of prediction probability in soft selection.
    """
    def __init__(self, num_classes, alpha, gamma, use_gpu=True):
        super(SS_LR, self).__init__()
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.sigmoid = nn.Sigmoid()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        eps = 1e-7
        probs = self.sigmoid(inputs)
        # Note that for the multi-label case, there is no need to convert scalar label to one-hot label.
        # Comment this line.
        targets = torch.zeros(probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        count_pos = targets.sum().cuda()
        count_neg = (1.0-targets).sum().cuda()
        if self.use_gpu: targets = targets.cuda()
        pred_p = torch.log(probs+eps).cuda()
        pred_n = (torch.pow(probs, self.gamma) * torch.log(1.0-probs+eps)).cuda()

        loss = (targets * pred_p).sum() / count_pos + self.alpha*(((1.0-targets) * pred_n).sum() / count_neg)

        return - 1.0 * loss



if __name__ == '__main__':
    pass