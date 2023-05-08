import os
import torch
import csv
import numpy as np

def check_make_dir(dir_path):
    if not os.path.exists(dir_path):
        print(f"{dir_path}is not exist, but now we make one.")
        os.makedirs(dir_path)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer"])


class write_csv():
    def __init__(self, save_root):
        super(write_csv, self)
        self.save_root = save_root

    def write_situation(self, header, situation, result_head=['index', 'name', 'acc', 'dice']):
        # header = ['train_path', 'val_path', 'test_path', 'model_name', 'channels', 'learning_rate',
        #           'repeat_time', 'num_epochs', 'batch_size',
        #           'train_sample_num', 'val_sample_num', 'time']
        # situation = [args.TRAIN_IMG_DIR, args.VAL_IMG_DIR, args.TEST_IMG_DIR, model_name,
        #              channels, args.learning_rate, args.repeat_time, args.num_epochs, args.batch_size,
        #              args.train_num, args.val_num, args.test_num, np.mean(np.array(time_avg))]
        # 记录平均的最终结果, np.mean(np.array(val_avg['con_mat']), axis=0)
        with open(f'{self.save_root}/situation.csv', 'w', encoding='utf-8', newline='') as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(header)
            writer.writerow(situation)
            writer.writerow(result_head)

    def write_result(self, head, results):
        # results = [('avg', 'train', np.mean(np.array(train_avg['acc']), axis=0), np.mean(np.array(train_avg['dice']), axis=0)),
        #            ('avg', 'val', np.mean(np.array(val_avg['acc']), axis=0), np.mean(np.array(val_avg['dice']), axis=0)),
        #            ('avg', 'test', np.mean(np.array(test_avg['acc']), axis=0), np.mean(np.array(test_avg['dice']), axis=0))]

        with open(f'{self.save_root}/situation.csv', 'w', encoding='utf-8', newline='') as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(head)
            for result in results:
                writer.writerow(result)
            # for i in range(0, len(train_avg['acc'])):
            #     writer.writerow((i + 1, 'train', np.array(train_avg['acc'][i]), np.array(train_avg['dice'][i])))
            #     writer.writerow((i + 1, 'val', np.array(val_avg['acc'][i]), np.array(val_avg['dice'][i])))
            #     writer.writerow((i + 1, 'test', np.array(test_avg['acc'][i]), np.array(test_avg['dice'][i])))
