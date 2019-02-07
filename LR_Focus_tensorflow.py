# =================================================================================
# Copyright 2019
# Single label multi-class image classification by deep logistic regression Authors.
# All Rights Reserved.

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =================================================================================

import numpy as np
import tensorflow as tf


def HS_LR(labels, logits, alpha, ratio):
    """ Implementation of Focus rectified logistic regression loss with Hard Selection(LR-HS)
        Args:
        labels: groundtruth label with shape [batch_size, num_classes]
                Note that for single label case, each row is a one-hot vector.
        logits: outputs of the last FC layer, with shape [batch_size, num_classes] (without normalisation).
        alpha: cost-sensitive hyperparameter in Eq. (5);
        ratio: m % in Eq. (5) for hard selection of most confusing negative classes.
    """
    eps = 1e-7
    prediction = tf.nn.sigmoid(logits)
    pos_log = tf.log(prediction + eps)
    neg_log = tf.log(1.0 - prediction + eps)
    # Note that: for single label case, count_pos is equal to batchsize;
    # while for the multi-label case, count_pos is equal to the overall number of positive labels per batch.
    count_pos = tf.reduce_sum(labels)
    topk = np.int64(labels.shape[1] * ratio)
    hard_neg_loss = -1.0 * (1.0 - labels) * neg_log
    topk_neg_loss = -1.0 * tf.nn.top_k(hard_neg_loss, topk)[0]
    loss = tf.reduce_sum(tf.multiply(labels, pos_log)) / tf.to_float(count_pos) + alpha * tf.reduce_mean(topk_neg_loss)

    return -1.0 * loss



def SS_LR(labels, logits, alpha, gamma):
    """ Implementation of Focus rectified logistic regression loss with Soft Selection(LR-SS).
        Args:
            labels: groundtruth label with shape [batch_size, num_classes]
                    Note that for single label case, each row is a one-hot vector.
            logits: outputs of the last FC layer, with shape [batch_size, num_classes] (without normalisation).
            alpha: cost-sensitive hyperparameter in Eq. (6);
            gamma: gamma in Eq. (6), the temperature of prediction probability in soft selection.
    """
    eps = 1e-7
    prediction = tf.nn.sigmoid(logits)
    # Note that: for single label case, count_pos is equal to batchsize;
    # while for the multi-label case, count_pos is equal to the overall number of positive labels per batch.
    count_pos = tf.reduce_sum(labels)
    count_neg = tf.reduce_sum(1-labels)
    pos_log = tf.log(prediction+eps)
    neg_log = tf.multiply(tf.pow((prediction), gamma), tf.log(1.0-prediction+eps))
    loss = tf.reduce_sum(tf.multiply(labels, pos_log))/tf.to_float(count_pos) \
           + alpha*tf.reduce_sum(tf.multiply((1.0-labels), neg_log))/tf.to_float(count_neg)

    return -1.0 * loss


