import torch
import numpy as np

eps = 1e-7
class calculate_pred_result():
    def __init__(self, logits, labels, num_classes=2, reduce_zero_label=False, ignored_label=None):
        super(calculate_pred_result, self)
        self.num_classes = num_classes
        self.logits = logits
        self.labels = labels
        self.pred_map = self.logits.argmax(1)
        self.correct_map = self.pred_map[self.pred_map == self.labels]
        # 原来的背景变为255
        if reduce_zero_label:
            self.reduce_zero()
        if ignored_label is not None:
            self.ignored_label = ignored_label
            self.ignored_label()
        self.calculate_for_each_class()
        self.calculate_for_all()

    # 如果存在不感兴趣的label
    def remove_ignored_label(self):
        self.interest_pos = (self.labels != self.ignored_label)
        self.labels = self.labels[self.interest_pos]
        self.pred_map = self.pred_map[self.interest_pos]
        self.correct_map = self.pred_map[self.pred_map == self.labels]
        # 变一维张量

    # 如果不想计算类别 0，此时的num_class也不包含类别0
    def reduce_zero(self):
        self.labels[self.labels == 0] = 255
        self.labels = self.labels - 1
        self.labels[self.labels == 254] = 255

    def calculate_for_each_class(self):
        self.num_true_each_class = torch.histc(
            self.labels.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)
        self.num_pred_each_class = torch.histc(
            self.pred_map.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)
        self.num_correct_each_class = torch.histc(
            self.correct_map.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)

    def calculate_for_all(self):
        self.num_correct = torch.sum(self.num_correct_each_class)
        self.num_true = torch.sum(self.num_true_each_class)
        self.num_preds = torch.sum(self.num_pred_each_class)
        self.inter = self.num_correct
        self.union = self.num_preds + self.num_true - self.num_correct


    def calculate_all_accuracy(self):
        return float(self.num_correct / (self.num_true + eps))

    def calculate_iou_all(self):
        return float(self.inter / (self.union + eps))

    # 计算每一类精度
    def calculate_each_class_accuracy(self):
        self.each_acc = self.num_correct_each_class / (self.num_true_each_class + eps)
        return self.each_acc

    # 计算每一类iou
    def calculate_each_class_iou(self):
        each_inter = self.num_correct_each_class
        each_union = self.num_pred_each_class + self.num_true_each_class - self.num_correct_each_class
        return each_inter / (each_union + eps)

    def calculate_precision(self):
        self.precision = self.num_correct / (self.num_preds + eps)
        return float(self.precision)

    def calculate_each_precision(self):
        self.each_precision = self.num_correct_each_class / (self.num_pred_each_class + eps)
        return self.each_precision


    def calculate_recall(self):
        self.recall = self.num_correct / (self.num_true + eps)
        return float(self.recall)

    def calculate_each_recall(self):
        self.each_recall = self.num_correct_each_class / (self.num_true_each_class + eps)
        return float(self.each_recall)

    def caculate_each_f_score(self, beta=1):
        self.each_score = (1 + beta**2) * (self.each_precision * self.each_recall) / (
            (beta**2 * self.each_precision) + self.each_recall + eps)
        return float(self.each_score)

    def calculate_f_score(self, beta=1):
        """calculate the f-score value.

        Args:
            precision (float | torch.Tensor): The precision value.
            recall (float | torch.Tensor): The recall value.
            beta (int): Determines the weight of recall in the combined score.
                Default: False.

        Returns:
            [torch.tensor]: The f-score value.
        """
        score = (1 + beta**2) * (self.precision * self.recall) / (
            (beta**2 * self.precision) + self.recall + eps)
        return float(score)



class epoch_compare():
    def __init__(self, max_epoch_stop=100):
        super(epoch_compare, self)
        self.min_loss = 100
        self.epoch_accumulate = 0
        self.max_epoch_stop = max_epoch_stop
        self.stop_flag = False

        self.max_acc = 0
        self.max_acc_epoch = 0
        self.max_iou = 0
        self.max_iou_epoch = 0

    def compare_loss(self, loss_this_epoch):
        if loss_this_epoch < self.min_loss:
            self.min_loss = loss_this_epoch
            self.epoch_accumulate = 0
        else:
            self.epoch_accumulate += 1
        if self.epoch_accumulate >= self.max_epoch_stop:
            self.stop_flag = True
        return self.stop_flag

    def compare_acc(self, acc_this_epoch, epoch):
        if acc_this_epoch > self.max_acc:
            self.max_acc = acc_this_epoch
            self.max_acc_epoch = epoch

    def compare_iou(self, iou_this_epoch, epoch):
        if iou_this_epoch > self.max_acc:
            self.max_iou = iou_this_epoch
            self.max_iou_epoch = epoch
