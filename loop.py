import itertools
from configs import Config
from train import train
import os
from time import localtime, strftime
import torch
import numpy as np
import xlsxwriter
from util import send_email
import traceback


def get_filename(base_filename, idx):
    return base_filename % idx


def main():
    try:
        ###############################################################
        # Update this box for every run - note the boolean args below
        scales = [2]
        exp_name = 'GBP_reg_lr_step=5'
        for scale in scales:
            hyper_params = {
                'files': range(1, 100 + 1),
                'dataset': ['DIV2K'],     # AIM/BSD100/urban/set5/set14/DIV2K/REAL/NEW_DATA_NF/anat/OTHER/BIC
            }

            flags = []
        ###############################################################
            dataset = hyper_params['dataset'][0]
            experiment_name = 'X%d_' % scale + exp_name
            output_dir = '/home/sefibe/data/kernelGAN_results/'
            flags += [] if scale == 2 else ['analytic_sf']
            if 'DIV2K' in dataset:
                base_dir = 'training_data/%s' % dataset
                input_filename = os.path.join(base_dir, 'lr_x%d' % scale, 'img_%d.png')
                input_gt_filename = os.path.join(base_dir, 'gt', 'img_%d_gt.png')
                kernel_gt_filename = os.path.join(base_dir, 'gt_k_x%d' % scale, 'kernel_%d.mat')
                kernel_tomer_filename = os.path.join(base_dir, 'tomer_k_x%d' % scale, 'kernel_%d.mat')
            elif dataset == 'REAL':
                base_dir = 'training_data/real_examples'
                input_filename = os.path.join(base_dir, 'img_%d.png')
                input_gt_filename = os.path.join(base_dir, 'img_%d_gt.png')
                kernel_gt_filename = os.path.join(base_dir, 'kernel_%d.mat')
                kernel_tomer_filename = os.path.join(base_dir, 'tomer_k_x%d' % scale, 'kernel_%d.mat')
                if 'real_image' not in flags:
                    flags += ['real_image']
            elif dataset == 'AIM':
                base_dir = 'training_data'
                input_filename = os.path.join(base_dir, 'ValidationSource', '0%d.png')
                input_gt_filename = os.path.join(base_dir, 'DIV2K/gt', 'img_%d_gt.png')
                kernel_gt_filename = os.path.join(base_dir, 'gt_k_x%d' % scale, 'kernel_%d.mat')
                kernel_tomer_filename = os.path.join(base_dir, 'tomer_k_x%d' % scale, 'kernel_%d.mat')
            elif dataset == 'BIC':
                base_dir = 'training_data/DIV2K'
                input_filename = os.path.join(base_dir, 'lr_x%d_bicubic' % scale, 'img_%d.png')
                input_gt_filename = os.path.join(base_dir, 'gt', 'img_%d_gt.png')
                kernel_gt_filename = os.path.join(base_dir, '%d')
                kernel_tomer_filename = os.path.join(base_dir, '%d')
            elif dataset in 'BSD100_set14_set5_urban':
                base_dir = 'training_data/%s' % dataset
                input_filename = os.path.join(base_dir, 'x%d' % scale, 'img_%d.png')
                # input_gt_filename = os.path.join(base_dir, 'gt', 'img_%d_gt.png')
                input_gt_filename = os.path.join(base_dir, 'gt', '%d.png')
                kernel_gt_filename = os.path.join(base_dir, '%d')
                kernel_tomer_filename = os.path.join(base_dir, '%d')
            else:
                base_dir = 'training_data/other'
                input_filename = os.path.join(base_dir, 'img_%d.png')
                input_gt_filename, kernel_gt_filename, kernel_tomer_filename = '%d', '%d', '%d'

            files_range = list(hyper_params.pop('files'))
            params_names = list(hyper_params.keys())
            params_ranges = [hyper_params[name] for name in params_names]

            for params_values in itertools.product(*params_ranges):
                folder_name = experiment_name
                args = sum([['--%s' % name, str(val)] for name, val in zip(params_names, params_values)], []) + ['--%s' % flag for flag in flags]
                print('\n\nSPECIAL CONFIGURATION:\nNAME:', experiment_name, '\n', args, '\n\n')
                for name, val in zip(params_names, params_values):
                    folder_name += '_%s=%s' % (name, str(val)) if name != 'dataset' else '_%s' % str(val)
                for flag in flags:
                    folder_name += '_' + flag
                loop_logger = LoopCheckpoint(output_dir + folder_name, folder_name, scale, dataset) if dataset != 'AIM' else AIMCheckpoint(output_dir + folder_name, folder_name, scale, dataset)
                for file_idx in files_range:
                    img_idx = file_idx if dataset != 'AIM' else file_idx + 800
                    args_with_files = args + ['--input_image_path', input_filename % img_idx,
                                              '--gt_image_path', input_gt_filename % file_idx,
                                              '--gt_kernel_path', kernel_gt_filename % file_idx,
                                              '--output_dir_path', output_dir + folder_name,
                                              '--tomer_kernel_path', kernel_tomer_filename % file_idx,
                                              '--name', 'Img_%d' % file_idx,
                                              '--file_idx', '%d' % file_idx,
                                              '--dataset', dataset]
                    print('input image: %s' % (input_filename % img_idx))
                    conf = Config().parse(map(str, args_with_files))
                    if conf.input_image_path is None:
                        continue
                    train(conf, loop_logger)

                perf, winners = loop_logger.done()
                print('\n\nFINISHED:\nNAME:', experiment_name, '\n', args, '\n\n')
                send_email(s='%s Finished' % experiment_name, m='\n %s winners=%s, performance=%s' % (folder_name, winners, perf))

    except KeyboardInterrupt:
        raise
    except Exception as e:
        traceback.print_exc()
        print('Stopped due to the error:', e)
        # send_email(s='%s Failed' % experiment_name, m='Error: %s' % repr(e))


class LoopCheckpoint:
    def __init__(self, output_path, name, scale, dataset):
        self.flag = (dataset == 'DIV2K')
        if self.flag:
            self.log = torch.Tensor()
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
            self.log_dir = output_path + '/' + name + '.txt'
            open_type = 'a' if os.path.exists(self.log_dir) else 'w'
            path_psnr = '/home/sefibe/Documents/KernelGAN/training_data/' + '%s' % dataset + '/sr_w_%s_k_x%d/%s_%s.npy'
            self.log_file = open(self.log_dir, open_type)
            self.log_file.write('-' * len(name) + '\n' + name + '\nIMG_NUM:\tGT_k\tTomer_k\tkgan_k\tWinner\n')
            self.xls_dir = output_path + '/' + name + strftime('_%b_%d_%H_%M_%S', localtime()) + '.xlsx'
            self.xls_file = xlsxwriter.Workbook(self.xls_dir)
            self.work_sheet = self.xls_file.add_worksheet()
            self.bold_format = self.xls_file.add_format({'bold': True})
            self.excel_title(name)
            # Load PSNR lists
            self.psnr_gt_no_bp, self.psnr_gt_zssr = np.load(path_psnr % ('gt', scale, 'PSNR', 'no_bp')), np.load(path_psnr % ('gt', scale, 'PSNR', 'zssr'))
            self.psnr_tomer_no_bp, self.psnr_tomer_zssr = np.load(path_psnr % ('tomer', scale, 'PSNR', 'no_bp')), np.load(path_psnr % ('tomer', scale, 'PSNR', 'zssr'))
            # Load MSE lists
            self.mse_gt_no_bp, self.mse_gt_zssr = np.load(path_psnr % ('gt', scale, 'mse', 'no_bp')), np.load(path_psnr % ('gt', scale, 'mse', 'zssr'))
            self.mse_tomer_no_bp, self.mse_tomer_zssr = np.load(path_psnr % ('tomer', scale, 'mse', 'no_bp')), np.load(path_psnr % ('tomer', scale, 'mse', 'zssr'))
            # calculate_ratio_tomer
            self.ratio_tomer_no_bp = [self.mse_gt_no_bp[idx] / self.mse_tomer_no_bp[idx] for idx in range(min(len(self.mse_gt_no_bp), len(self.mse_tomer_no_bp)))]
            self.ratio_tomer_zssr = [self.mse_gt_zssr[idx] / self.mse_tomer_zssr[idx] for idx in range(min(len(self.mse_gt_zssr), len(self.mse_tomer_zssr)))]
            self.results_zssr, self.results_no_bp = [], []

    def excel_title(self, name):
        cells = ['E1', 'J1', 'B2', 'G2', 'D2', 'D3', 'I2', 'I3', 'B4', 'C4', 'D4', 'E4', 'G4', 'H4', 'I4', 'J4', 'L2', 'L3', 'L4', 'M4', 'O3', 'O4', 'P4']
        titles = [name, name, 'NO_BP', 'ZSSR', 'Win', 'Mean', 'Win', 'Mean', 'GT_k', 'Tomer', 'kGAN', 'Diff', 'GT_k', 'Tomer', 'kGAN', 'Diff', 'MSE_Ratio', 'NO_BP', 'Tomer', 'kGAN', 'ZSSR', 'Tomer', 'kGAN']
        [self.work_sheet.write(cells[ind], titles[ind], self.bold_format) for ind in range(len(cells))]

    def write_results(self, idx, psnr_no_bp, psnr_zssr, mse_no_bp, mse_zssr):
        if not self.flag:
            return
        diff_no_bp, diff_zssr = psnr_no_bp-self.psnr_tomer_no_bp[idx], psnr_zssr-self.psnr_tomer_zssr[idx]
        self.results_no_bp.append(diff_no_bp)
        self.results_zssr.append(diff_zssr)
        winner_no_bp = '+%.2f' % diff_no_bp if diff_no_bp > 0 else '-%.2f' % -diff_no_bp
        winner_zssr = '+%.2f' % diff_zssr if diff_zssr > 0 else '-%.2f' % -diff_zssr
        verbose = 'img%d_%s\t%.2f\t%.2f\t%.2f\t%s\t'
        self.log_file.write(verbose % (idx, 'no_bp', self.psnr_gt_no_bp[idx], self.psnr_tomer_no_bp[idx], psnr_no_bp, winner_no_bp))
        self.log_file.write(verbose % (idx, 'zssr', self.psnr_gt_zssr[idx], self.psnr_tomer_zssr[idx], psnr_zssr, winner_zssr) + '\n')
        # Calculate MSE(gt) / MSE(KGAN)
        ratio_kgan_no_bp, ratio_kgan_zssr = self.mse_gt_no_bp[idx] / mse_no_bp, self.mse_gt_zssr[idx] / mse_zssr
        self.write_to_excel(idx, psnr_no_bp, psnr_zssr, diff_no_bp, diff_zssr, ratio_kgan_no_bp, ratio_kgan_zssr)

    def write_to_excel(self, idx, psnr_no_bp, psnr_zssr, diff_no_bp, diff_zssr, ratio_kgan_no_bp, ratio_kgan_zssr):
        self.work_sheet.write('A%d' % (idx + 4), int('%d' % idx))
        results = [self.psnr_gt_no_bp[idx], self.psnr_tomer_no_bp[idx], psnr_no_bp, diff_no_bp,
                   self.psnr_gt_zssr[idx], self.psnr_tomer_zssr[idx], psnr_zssr, diff_zssr,
                   self.ratio_tomer_no_bp[idx], ratio_kgan_no_bp,
                   self.ratio_tomer_zssr[idx], ratio_kgan_zssr]
        cells = 'BCDEGHIJLMOP'
        [self.work_sheet.write('%s%d' % (cells[ind], (idx + 4)), float('%.2f' % results[ind])) for ind in range(len(results))]

    def done(self):
        if not self.flag:
            return 0, 0
        winners_no_bp = sum([1 for x in self.results_no_bp if x >= 0])
        winners_zssr = sum([1 for x in self.results_zssr if x >= 0])
        num_imgs = len(self.results_zssr)
        if num_imgs == 0:
            return '0', '0/0'
        self.log_file.write('\n' + '-' * 90 +
                            '\nNo B.P. Winners = %d/%d = %d %%  Mean = %.2f dB\t\tZSSR Winners = %d/%d = %d %%  Mean = %.2f dB' %
                            (winners_no_bp, num_imgs, int(100 * winners_no_bp / num_imgs), sum(self.results_no_bp) / num_imgs,
                             winners_zssr, num_imgs, int(100 * winners_zssr / num_imgs), sum(self.results_zssr) / num_imgs))
        self.results_no_bp, self.results_zssr = np.array(self.results_no_bp), np.array(self.results_zssr)
        self.log_file.write('\n\nOnly non-catastrophes (=diff<2): \t No B.P. %d/%d Mean = %.2f dB\t\tZSSR %d/%d Mean = %.2f dB' %
                            (len(self.results_no_bp[np.abs(self.results_no_bp) < 2]), num_imgs,
                             self.results_no_bp[np.abs(self.results_no_bp) < 2].sum() / len(self.results_no_bp[np.abs(self.results_no_bp) < 2]),
                             len(self.results_zssr[np.abs(self.results_zssr) < 2]), num_imgs,
                             self.results_zssr[np.abs(self.results_zssr) < 2].sum() / len(self.results_no_bp[np.abs(self.results_no_bp) < 2])))
        self.log_file.close()

        # All results
        self.work_sheet.write('E2', '%d/%d' % (winners_no_bp, len(self.results_no_bp)))
        self.work_sheet.write('E3', float('%.2f' % (sum(self.results_no_bp)/len(self.results_no_bp))))
        self.work_sheet.write('J2', '%d/%d' % (winners_zssr, len(self.results_zssr)))
        self.work_sheet.write('J3', float('%.2f' % (sum(self.results_zssr)/len(self.results_zssr))))

        self.xls_file.close()

        return '%.2f' % (sum(self.results_zssr)/len(self.results_zssr)), '%d/%d' % (winners_zssr, len(self.results_zssr))


class AIMCheckpoint:
    def __init__(self, output_path, name, scale, dataset):
        self.flag = dataset == 'AIM'
        if self.flag:
            self.log = torch.Tensor()
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
            self.xls_dir = output_path + '/' + name + strftime('_%b_%d_%H_%M_%S', localtime()) + '.xlsx'
            self.xls_file = xlsxwriter.Workbook(self.xls_dir)
            self.work_sheet = self.xls_file.add_worksheet()
            self.bold_format = self.xls_file.add_format({'bold': True})
            self.excel_title(name)
            self.results_zssr = []

    def excel_title(self, name):
        cells = ['B2', 'C1']
        titles = [name, 'PSNR']
        [self.work_sheet.write(cells[ind], titles[ind], self.bold_format) for ind in range(len(cells))]

    def write_results(self, idx, null1, psnr_zssr, null2, null3):
        if not self.flag:
            return
        self.results_zssr.append(psnr_zssr)
        self.work_sheet.write('B%d' % (idx + 2), idx)
        self.work_sheet.write('C%d' % (idx + 2), float('%.2f' % psnr_zssr))

    def done(self):
        if not self.flag:
            return 0, 0
        num_imgs = len(self.results_zssr)
        if num_imgs == 0:
            return '0', '0/0'
        self.results_zssr = np.array(self.results_zssr)
        # All results
        self.work_sheet.write('D2', 'Avg')
        self.work_sheet.write('C2', float('%.2f' % (sum(self.results_zssr)/len(self.results_zssr))))
        self.xls_file.close()

        return '%.2f' % (sum(self.results_zssr)/len(self.results_zssr)), ''


if __name__ == '__main__':
    main()
