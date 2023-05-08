from wh_frame.parameters import class_to_keep, class_to_remove
from wh_frame.parameters import train_map, train_label, train_num, val_map, val_label, val_num
from wh_frame.parameters import map_seffix, label_seffix, in_h, in_w
from wh_frame.parameters import augmentation_methods
from wh_frame.parameters import model_name, in_channels, out_channels, muti_class
from wh_frame.parameters import LOAD_TRAIN_MODEL, LOAD_TRAIN_PATH
from wh_frame.parameters import num_workers, pin_memory, gpu
from wh_frame.parameters import lr_start, loss_name, loss_stop_epoch
from wh_frame.parameters import optimizer_name, scheduler_name
from wh_frame.parameters import batch_size, epoch_max, epoch_min, repeat_times
from wh_frame.parameters import save_root, result_dir, model_save_dir, tensorboard_log_dir, mask_val_save_dir


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


def train(index, epoch, model, loader, optimizer, muti_class=False):

    if muti_class:
        result_name = ['acc', 'iou', 'each_acc', 'each_iou', 'loss']
    else:
        result_name = ['acc', 'iou', 'loss']
    result_record = dict.fromkeys(result_name)
    result_avg = dict.fromkeys(result_name)
    # 把dict中None 转为 list
    for i in result_record:
        result_record[i] = []
    # start_t = t = time.time()
    model.train()
    loop = tqdm(loader)
    for it, (images, labels, names) in enumerate(loop):
        images = Variable(images).cuda(gpu)
        labels = Variable(labels).cuda(gpu)
        if gpu >= 0:
            images = images.cuda(gpu)
            images = images.float()
            labels = labels.cuda(gpu)
            labels = labels.long()
        logits = model(images)
        loss = eval(loss_name)(logits=logits, labels=labels)
        loss = loss.loss_function()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scaler.update()

        loss_value = loss.detach()
        if np.isnan(np.asarray(loss_value.cpu())).any():
            breakpoint()
        # update tqdm loop
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
        result_record['loss'].append(loss_value.cpu().numpy())

    for i in result_record:
        temp = np.asarray(result_record[i])
        result_avg[i] = np.mean(temp, axis=0)

    for param_group in optimizer.param_groups:
        lr_this_epo = param_group['lr']
        print('lr %s' % (lr_this_epo))
        break

    with open(log_train_file, 'a') as appender:
        appender.write(f'index: {index}, epoch: {epoch}')
        for name in result_name:
            content = name + ':' + str(result_avg[name])
            print(content)
            appender.write(content)
            appender.write('  ')
        appender.write('\n')

    return result_avg


def validation(index, epoch, model, loader, muti_class):

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

    with open(log_val_file, 'a') as appender:
        appender.write(f'index: {index}, epoch: {epoch}')
        for name in result_name:
            content = name + ':' + str(result_avg[name])
            print(content)
            appender.write(content)
            appender.write('  ')
        appender.write('\n')

    return result_avg


def main():
    val_dataset = forest(image_dir=val_map, mask_dir=val_label,
                         transform=augmentation_methods, num=val_num,
                         label_to_keep=class_to_keep, label_to_remove=class_to_remove)
    train_dataset = forest(image_dir=train_map, mask_dir=train_label,
                           transform=augmentation_methods, num=train_num,
                           label_to_keep=class_to_keep, label_to_remove=class_to_remove)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              drop_last=True)
    print("train data has been loaded!")
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            drop_last=True
                            )
    print("val data has been loaded!")
    train_loader.n_iter = len(train_loader)
    val_loader.n_iter = len(val_loader)

    for index in range(0, repeat_times):
        model_name_all = 'model_dir.' + model_name

        model = eval(model_name_all)(in_channels, out_channels)
        model.to(DEVICE)

        optimizer = eval(optimizer_name)
        scheduler = eval(scheduler_name)


        if LOAD_TRAIN_MODEL:
            load_checkpoint(torch.load(LOAD_TRAIN_PATH), model, optimizer)
        else:
            model.initialize_weights()

        writer = SummaryWriter(log_dir=tensorboard_log_dir)
        epoch_compare_all = epoch_compare(loss_stop_epoch)

        for epoch in range(epoch_min, epoch_max + 1):

            print('\n| index: %s epoch %s begin...' % (index, epoch))
            t_result = train(index, epoch, model, train_loader, optimizer, muti_class)
            v_result = validation(index, epoch, model, val_loader, muti_class)
            # record the score to tensorboard
            writer.add_scalars('train_acc', {save_root: float(t_result['acc'])}, epoch)
            writer.add_scalars('train_loss', {save_root: float(t_result['loss'])}, epoch)
            writer.add_scalars('val_acc', {save_root: float(v_result['acc'])}, epoch)
            writer.add_scalars('val_loss', {save_root: float(v_result['loss'])}, epoch)
            # lr
            scheduler.step(t_result['loss'])
            # for validation
            stop_flag = epoch_compare_all.compare_loss(v_result['loss'])
            epoch_compare_all.compare_acc(v_result['acc'], epoch)
            epoch_compare_all.compare_iou(v_result['iou'], epoch)

            # save checkpoint
            print('| saving check point model file... ', end='')
            checkpoint_epoch_name = model_name +'_epo' + str(epoch) +'_viou' + str(v_result['iou']) + '_tacc' + str(t_result['acc']) +'_vacc' +str(v_result['acc']) +'.pth'
            save_path = os.path.join(model_save_dir, checkpoint_epoch_name)
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            save_checkpoint(checkpoint, save_path)

            if stop_flag:
                break

        # record results
        head = ['index', 'name', 'acc', 'iou', 'loss', 'max_acc', 'max_acc_epoch',
                'max_iou', 'max_iou_epoch', 'min_loss', 'epoch']
        result = [(index, 'train', t_result['acc'], t_result['iou'], t_result['loss']),
                  (index, 'val', v_result['acc'], v_result['iou'], v_result['loss'],
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
    param = [train_map, train_label, train_num, val_map, val_label, val_num,
             map_seffix, label_seffix, in_h, in_w, augmentation_methods,
             model_name, in_channels, out_channels,
             LOAD_TRAIN_MODEL, LOAD_TRAIN_PATH,
             num_workers, pin_memory, gpu,
             lr_start, loss_name, loss_stop_epoch,
             optimizer_name, scheduler_name,
             batch_size, epoch_max, epoch_min, repeat_times,
             save_root, result_dir, model_save_dir, tensorboard_log_dir, mask_val_save_dir,
             ]

    param_name = ['train_map', ' train_label', ' train_num', ' val_map', ' val_label', ' val_num',
                'map_seffix', ' label_seffix', ' in_h', ' in_w', ' augmentation_methods',
                'model_name', ' in_channels', ' out_channels',
                'LOAD_TRAIN_MODEL', ' LOAD_TRAIN_PATH',
                'num_workers', ' pin_memory', ' gpu',
                'lr_start', ' loss_name', ' loss_stop_epoch',
                'optimizer_name', ' scheduler_name',
                'batch_size', ' epoch_max', ' epoch_min', ' repeat_times',
                'save_root', ' result_dir', ' model_save_dir', ' tensorboard_log_dir', ' mask_val_save_dir',
               ]

    check_make_dir(save_root)
    check_make_dir(result_dir)
    check_make_dir(model_save_dir)
    check_make_dir(tensorboard_log_dir)
    check_make_dir(mask_val_save_dir)
    write_this_csv = write_csv(save_root=save_root)
    write_this_csv.write_situation(param_name, param, result_head=result_head)
    log_train_file = os.path.join(save_root, 'log_train.txt')
    log_val_file = os.path.join(save_root, 'log_val.txt')

    print('| training %s on GPU #%d with pytorch' % (model_name, gpu))
    print('| from epoch %d / %s' % (epoch_min, epoch_max))
    print('| model will be saved in: %s' % model_save_dir)

    main()
