from wh_frame.parameters import class_to_keep, class_to_remove
from wh_frame.parameters import test_map, test_label, test_num
from wh_frame.parameters import map_seffix, label_seffix, in_h, in_w
from wh_frame.parameters import augmentation_methods
from wh_frame.parameters import model_name, in_channels, out_channels, muti_class
from wh_frame.parameters import LOAD_TEST_PATH
from wh_frame.parameters import num_workers, pin_memory, gpu
from wh_frame.parameters import lr_start, loss_name, loss_stop_epoch
from wh_frame.parameters import optimizer_name, scheduler_name
from wh_frame.parameters import batch_size, epoch_max, epoch_min, repeat_times
from wh_frame.parameters import save_root, result_dir, model_save_dir, tensorboard_log_dir, mask_test_save_dir


from torch.utils.tensorboard import SummaryWriter
from utils import check_make_dir, load_checkpoint, save_checkpoint, write_csv
from result_fun import calculate_pred_result, epoch_compare
from dataset import forest
import time
import numpy as np
import os
import torch
import model_dir

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from collections import OrderedDict
from loss.lossfunction import IoULoss, CrossEntropy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def test(index, epoch, model, loader, muti_class):

    if muti_class:
        result_name = ['acc', 'iou', 'loss', 'precision', 'recall', 'f1', 'each_acc', 'each_iou']
    else:
        result_name = ['acc', 'iou', 'loss', 'precision', 'recall', 'f1']
    result_record = dict.fromkeys(result_name)
    result_avg = dict.fromkeys(result_name)
    # 把dict中None 转为 list
    for i in result_record:
        result_record[i] = []
    start_t = time.time()
    model.eval()
    loop = tqdm(loader)
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(loop):
            images = Variable(images)
            labels = Variable(labels)
            if gpu >= 0:
                images = images.cuda(gpu)
                images = images.float()
                labels = labels.cuda(gpu)
                labels = labels.long()

                # print(np.unique(labels.cpu().numpy()))
                # print(names)
            logits = model(images)
            loss = eval(loss_name)(logits=logits, labels=labels)
            loss = loss.loss_function()

            loss_value = loss.detach()
            if np.isnan(np.asarray(loss_value.cpu())).any():
                breakpoint()
            loop.set_postfix(loss=loss)
            calculate_result = calculate_pred_result(logits=logits, labels=labels, num_classes=out_channels)

            if muti_class:
                each_acc = calculate_result.calculate_each_class_accuracy()
                each_iou = calculate_result.calculate_each_class_iou()
                result_record['each_acc'].append(each_acc.cpu().numpy())
                result_record['each_iou'].append(each_iou.cpu().numpy())
            acc = calculate_result.calculate_all_accuracy()
            result_record['acc'].append(acc)
            iou = calculate_result.calculate_iou_all()
            result_record['iou'].append(iou)
            precision = calculate_result.calculate_precision()
            result_record['precision'].append(precision)
            recall = calculate_result.calculate_recall()
            result_record['recall'].append(recall)
            f_1 = calculate_result.calculate_f_score()
            result_record['f1'].append(f_1)
            result_record['loss'].append(loss_value.cpu().numpy())


    for i in result_record:
        temp = np.asarray(result_record[i])
        result_avg[i] = np.mean(temp, axis=0)

    with open(log_test_file, 'a') as appender:
        appender.write(f'index: {index}, epoch: {epoch}')
        for name in result_name:
            content = name + ':' + str(result_avg[name])
            print(content)
            appender.write(content)
            appender.write('  ')
        appender.write('\n')

    return result_avg



def main():
    test_dataset = forest(image_dir=test_map, mask_dir=test_label,
                         transform=augmentation_methods, num=test_num,
                         label_to_keep=class_to_keep, label_to_remove=class_to_remove)


    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            drop_last=True
                            )
    print("test data has been loaded!")

    test_loader.n_iter = len(test_loader)

    for index in range(0, 1):
        model_name_all = 'model_dir.' + model_name

        model = eval(model_name_all)(in_channels, out_channels)
        model.to(DEVICE)

        optimizer = eval(optimizer_name)
        scheduler = eval(scheduler_name)

        load_checkpoint(torch.load(LOAD_TEST_PATH), model, optimizer)


        writer = SummaryWriter(log_dir=tensorboard_log_dir)
        epoch_compare_all = epoch_compare(loss_stop_epoch)

        for epoch in range(0, 1):

            print('\n| index: %s epoch %s begin...' % (index, epoch))

            t_result = test(index, epoch, model, test_loader, muti_class)
            # record the score to tensorboard

            writer.add_scalars('test_acc', {save_root: float(t_result['acc'])}, epoch)
            writer.add_scalars('test_loss', {save_root: float(t_result['loss'])}, epoch)
            # lr
            scheduler.step(t_result['loss'])
            # for validation
            stop_flag = epoch_compare_all.compare_loss(t_result['loss'])
            epoch_compare_all.compare_acc(t_result['acc'], epoch)
            epoch_compare_all.compare_iou(t_result['iou'], epoch)

        # record results
        head = ['index', 'name', 'acc', 'iou', 'loss', 'max_acc', 'max_acc_epoch',
                'max_iou', 'max_iou_epoch', 'min_loss', 'epoch']
        result = [
                  (index, 'test', t_result['acc'], t_result['iou'], t_result['loss'],
                   epoch_compare_all.max_acc, epoch_compare_all.max_acc_epoch,
                   epoch_compare_all.max_iou, epoch_compare_all.max_iou_epoch,
                   epoch_compare_all.min_loss, epoch)]
        write_this_csv.write_result(head=head, results=result)
    writer.close()

if __name__ == '__main__':
    result_head = ['index', 'name', 'acc', 'iou', 'loss',
                   'max_acc', 'max_acc_epoch' ,
                   'max_iou', 'max_iou_epoch',
                   'min_loss', 'epoch_to_stop'
                   ]
    param = [test_map, test_label, test_num, mask_test_save_dir]

    param_name = ['test_map', 'test_label', 'test_num'' mask_test_save_dir',
               ]
    test_save_root = os.path.join(save_root, 'test')
    check_make_dir(test_save_root)
    write_this_csv = write_csv(save_root=test_save_root)
    write_this_csv.write_situation(param_name, param, result_head=result_head)
    log_test_file = os.path.join(save_root, 'log_test.txt')

    print('| testing %s on GPU #%d with pytorch' % (model_name, gpu))
    print('| from epoch %d / %s' % (epoch_min, epoch_max))
    print('| model will be loaded from: %s' % LOAD_TEST_PATH)

    main()
