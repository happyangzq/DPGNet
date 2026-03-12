import os
import pdb
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import pickle
import datetime
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from metrics.base_metrics_class import Recorder
from torch.optim.swa_utils import AveragedModel
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from trainer.metrics.utils import get_test_metrics
from torch.nn import DataParallel
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

FFpp_pool = ['FaceForensics++']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer(object):
    def __init__(
            self,
            config,
            model,
            optimizer,
            scheduler,
            logger,
            metric_scoring='auc',
            time_now=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
            swa_model=None
    ):
        if config is None or model is None or optimizer is None or logger is None:
            raise ValueError("config, model, optimizier, logger, and tensorboard writer must be implemented")
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.swa_model = swa_model
        self.writers = {}
        self.logger = logger
        self.metric_scoring = metric_scoring
        self.best_metrics_all_time = defaultdict(
            lambda: defaultdict(lambda: float('-inf') if self.metric_scoring != 'eer' else float('inf'))
        )
        self.timenow = time_now
        self.speed_up()
        if 'task_target' not in config:
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + '_' + self.timenow
            )
        else:
            task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + task_str + '_' + self.timenow
            )
        os.makedirs(self.log_dir, exist_ok=True)

    def get_writer(self, phase, dataset_key, metric_key):
        writer_key = f"{phase}-{dataset_key}-{metric_key}"
        if writer_key not in self.writers:
            writer_path = os.path.join(self.log_dir, phase, dataset_key, metric_key, "metric_board")
            os.makedirs(writer_path, exist_ok=True)
            self.writers[writer_key] = SummaryWriter(writer_path)
        return self.writers[writer_key]

    def speed_up(self):
        self.model.to(device)
        self.model.device = device
        if self.config['ddp'] == True:
            num_gpus = torch.cuda.device_count()
            print(f'avai gpus: {num_gpus}')
            self.model = DDP(self.model, device_ids=[self.config['local_rank']], find_unused_parameters=True,
                             output_device=self.config['local_rank'])

    def DP_speed_up(self):
        self.model.device = device
        self.model.to(device)
        self.model = DataParallel(self.model, device_ids=[0,1])

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            suffix = model_path.split('.')[-1]
            if suffix == 'p':
                self.model.load_state_dict(saved.state_dict())
            else:
                self.model.load_state_dict(saved)
            self.logger.info('Model found in {}'.format(model_path))
        else:
            raise NotImplementedError("=> no model found at '{}'".format(model_path))

    def save_ckpt(self, phase, dataset_key, ckpt_info=None):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"ckpt_best.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        if self.config['ddp'] == True:
            torch.save(self.model.state_dict(), save_path)
        else:
            if 'svdd' in self.config['model_name']:
                torch.save({'R': self.model.R, 'c': self.model.c, 'state_dict': self.model.state_dict()}, save_path)
            else:
                torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Checkpoint saved to {save_path}, current ckpt is {ckpt_info}")

    def save_swa_ckpt(self):
        save_dir = self.log_dir
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"swa.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        torch.save(self.swa_model.state_dict(), save_path)
        self.logger.info(f"SWA Checkpoint saved to {save_path}")

    def save_feat(self, phase, fea, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        features = fea
        feat_name = f"feat_best.npy"
        save_path = os.path.join(save_dir, feat_name)
        np.save(save_path, features)
        self.logger.info(f"Feature saved to {save_path}")

    def save_data_dict(self, phase, data_dict, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'data_dict_{phase}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(data_dict, file)
        self.logger.info(f"data_dict saved to {file_path}")

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'metric_dict_best.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(metric_one_dataset, file)
        self.logger.info(f"Metrics saved to {file_path}")

    def visualize_tsne(self, epoch, iteration, dataset_name, features, labels, preds, save_path):
        """
        t-SNE 可视化（按真实标签显示 real 和 fake 类别，使用鲜明颜色）
        """
        # 根据真实标签分配类别
        categories = ['fake' if label == 1 else 'real' for label in labels]
        
        # 转换为 numpy 数组
        features = np.array(features)
        categories = np.array(categories)
        
        # t-SNE 降维
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
        tsne_features = tsne.fit_transform(features)
        
        # 绘制散点图
        plt.figure(figsize=(10, 8))
        unique_categories = ['real', 'fake']
        colors = ['#0000FF', '#FF0000']  # 蓝色 (real) 和红色 (fake)
        for i, category in enumerate(unique_categories):
            mask = categories == category
            plt.scatter(
                tsne_features[mask, 0],
                tsne_features[mask, 1],
                c=colors[i],
                label=category,
                s=60,  # 点的大小
                alpha=0.6,  # 透明度
                edgecolors='dimgray',  # 点边缘色
                linewidths=0.5
            )
        
        # 图表样式优化
        confi = self.model.module.confi if type(self.model) is DDP else self.model.confi
        plt.title(f't-SNE 可视化 - {dataset_name} (Epoch {epoch}, Iter {iteration}, Confi {confi:.3f})', fontsize=14)
        plt.xlabel('t-SNE 维度 1', fontsize=11)
        plt.ylabel('t-SNE 维度 2', fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=9)
        
        # 保存图像（高分辨率）
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"t-SNE 图（按真实标签，鲜明颜色）保存至 {save_path}")

    def train_step(self, data_dict, epoch, total_steps=None):
        if self.config['optimizer']['type'] == 'sam':
            for i in range(2):
                predictions = self.model(data_dict)
                losses = self.model.get_losses(data_dict, predictions,epoch)
                if i == 0:
                    pred_first = predictions
                    losses_first = losses
                self.optimizer.zero_grad()
                losses['overall'].backward()
                if i == 0:
                    self.optimizer.first_step(zero_grad=True)
                else:
                    self.optimizer.second_step(zero_grad=True)
            return losses_first, pred_first
        else:
            self.optimizer.zero_grad()
            predictions = self.model(data_dict,epoch)
           # if epoch ==1:
                #pdb.set_trace()
            if type(self.model) is DDP:
                losses = self.model.module.get_losses(data_dict, predictions,epoch,total_steps=total_steps)
            else:
                losses = self.model.get_losses(data_dict, predictions,epoch,total_steps=total_steps)
            losses['overall'].backward()
            self.optimizer.step()
            return losses, predictions

    def train_epoch(self, epoch, train_data_loader, test_data_loaders=None):
        self.logger.info("===> Epoch[{}] start!".format(epoch))
        train_loader_x, train_loader_u = train_data_loader
        if epoch >= 1:
            times_per_epoch = 2
        else:
            times_per_epoch = 1
        test_step = len(train_loader_x) // times_per_epoch
        step_cnt = epoch * len(train_loader_x)

        train_recorder_loss = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)
        total_target_samples = 0
        total_source_samples = 0
        self.model.reset_step_counter()  # 重置 batch 计数器
        for iteration, (data_x, data_u) in tqdm(enumerate(zip(train_loader_x, train_loader_u)), total=len(train_loader_x)):
            self.setTrain()
            data_dict = {
                'image': data_x['image'].to(device),
                'label': data_x['label'].to(device),
                'image_u': data_u['image'].to(device),
                'label_u': data_u['label']
            }
            total_target_samples += data_u['image'].size(0)
            total_source_samples += data_x['image'].size(0)

            losses, predictions = self.train_step(data_dict, epoch,total_steps=len(train_loader_x))
            if 'SWA' in self.config and self.config['SWA'] and epoch > self.config['swa_start']:
                self.swa_model.update_parameters(self.model)

            if type(self.model) is DDP:
                batch_metrics = self.model.module.get_train_metrics(data_dict, predictions)
            else:
                batch_metrics = self.model.get_train_metrics(data_dict, predictions)

            for name, value in batch_metrics.items():
                train_recorder_metric[name].update(value)
            for name, value in losses.items():
                train_recorder_loss[name].update(value)

            if iteration % 300 == 0:
                if 'SWA' in self.config and self.config['SWA'] and (epoch > self.config['swa_start'] or self.config['dry_run']):
                    self.scheduler.step()
                loss_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_loss.items():
                    v_avg = v.average()
                    if v_avg is None:
                        loss_str += f"training-loss, {k}: not calculated"
                        continue
                    loss_str += f"training-loss, {k}: {v_avg}    "
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_loss/{k}', v_avg, global_step=step_cnt)
                self.logger.info(loss_str)
                metric_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_metric.items():
                    v_avg = v.average()
                    if v_avg is None:
                        metric_str += f"training-metric, {k}: not calculated    "
                        continue
                    metric_str += f"training-metric, {k}: {v_avg}    "
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_metric/{k}', v_avg, global_step=step_cnt)
                self.logger.info(metric_str)
                confi = self.model.confi_max
                self.logger.info(f"Iter: {step_cnt}, Dynamic Threshold: {confi:.3f}")
                for name, recorder in train_recorder_loss.items():
                    recorder.clear()
                for name, recorder in train_recorder_metric.items():
                    recorder.clear()

            if (step_cnt + 1) % test_step == 0:
                if test_data_loaders is not None:
                    self.logger.info("===> Test start!")
                    test_iter = (step_cnt + 1) // test_step
                    test_best_metric = self.test_epoch(epoch, test_iter, test_data_loaders, step_cnt)
            step_cnt += 1
        self.logger.info(f"Total target domain samples used for pseudo-label loss in epoch {epoch}: {total_target_samples}")
        self.logger.info(f"Total source domain samples used for train in epoch {epoch}: {total_source_samples}")

        return test_best_metric

    def get_respect_acc(self, prob, label):
        pred = np.where(prob > 0.8, 1, 0)
        judge = (pred == label)
        
        # 根据真实标签值筛选样本，而不是位置
        real_indices = (label == 0)
        fake_indices = (label == 1)
        
        # 计算真实样本准确率
        real_count = np.sum(real_indices)
        acc_real = np.sum(judge[real_indices]) / real_count if real_count > 0 else 0
        
        # 计算伪造样本准确率
        fake_count = np.sum(fake_indices)
        acc_fake = np.sum(judge[fake_indices]) / fake_count if fake_count > 0 else 0
        
        return acc_real, acc_fake

    def test_one_dataset(self, data_loader,epoch):
        test_recorder_loss = defaultdict(Recorder)
        prediction_lists = []
        label_lists = []
        feature_lists = []
        for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
            if 'label_spe' in data_dict:
                data_dict.pop('label_spe')
            data_dict['label'] = torch.where(data_dict['label'] != 0, 1, 0)
            for key in data_dict.keys():
                if data_dict[key] is not None:
                    data_dict[key] = data_dict[key].cuda()
            predictions = self.inference(data_dict,epoch)
            label_lists += list(data_dict['label'].cpu().detach().numpy())
            prediction_lists += list(predictions['prob'].cpu().detach().numpy())
            feature_lists += list(predictions['features'].cpu().detach().numpy())
            if type(self.model) is not AveragedModel:
                if type(self.model) is DDP:
                    losses = self.model.module.get_losses(data_dict, predictions,epoch)
                else:
                    losses = self.model.get_losses(data_dict, predictions,epoch)
                for name, value in losses.items():
                    test_recorder_loss[name].update(value)
        return test_recorder_loss, np.array(prediction_lists), np.array(label_lists), np.array(feature_lists)

    def save_best(self, epoch, iteration, step, losses_one_dataset_recorder, key, metric_one_dataset):
        best_metric = self.best_metrics_all_time[key].get(self.metric_scoring,
                                                          float('-inf') if self.metric_scoring != 'eer' else float('inf'))
        improved = (metric_one_dataset[self.metric_scoring] > best_metric) if self.metric_scoring != 'eer' else (
                metric_one_dataset[self.metric_scoring] < best_metric)
        if improved:
            self.best_metrics_all_time[key][self.metric_scoring] = metric_one_dataset[self.metric_scoring]
            if key == 'avg':
                self.best_metrics_all_time[key]['dataset_dict'] = metric_one_dataset['dataset_dict']
            if self.config['save_ckpt'] and key not in FFpp_pool:
                self.save_ckpt('test', key, f"{epoch}+{iteration}")
            self.save_metrics('test', metric_one_dataset, key)
        if losses_one_dataset_recorder is not None:
            loss_str = f"dataset: {key}    step: {step}    "
            for k, v in losses_one_dataset_recorder.items():
                writer = self.get_writer('test', key, k)
                v_avg = v.average()
                if v_avg is None:
                    print(f'{k} is not calculated')
                    continue
                writer.add_scalar(f'test_losses/{k}', v_avg, global_step=step)
                loss_str += f"testing-loss, {k}: {v_avg}    "
            self.logger.info(loss_str)
        metric_str = f"dataset: {key}    step: {step}    "
        for k, v in metric_one_dataset.items():
            if k == 'pred' or k == 'label' or k == 'dataset_dict':
                continue
            metric_str += f"testing-metric, {k}: {v}    "
            writer = self.get_writer('test', key, k)
            writer.add_scalar(f'test_metrics/{k}', v, global_step=step)
        if 'pred' in metric_one_dataset:
            acc_real, acc_fake = self.get_respect_acc(metric_one_dataset['pred'], metric_one_dataset['label'])
            metric_str += f'testing-metric, acc_real:{acc_real}; acc_fake:{acc_fake}'
            writer.add_scalar(f'test_metrics/acc_real', acc_real, global_step=step)
            writer.add_scalar(f'test_metrics/acc_fake', acc_fake, global_step=step)
        self.logger.info(metric_str)

    def test_epoch(self, epoch, iteration, test_data_loaders, step):
        self.setEval()
        losses_all_datasets = {}
        metrics_all_datasets = {}
        best_metrics_per_dataset = defaultdict(dict)
        avg_metric = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0, 'video_auc': 0, 'dataset_dict': {}}
        keys = test_data_loaders.keys()
        for key in keys:
            data_dict = test_data_loaders[key].dataset.data_dict
            self.save_data_dict('test', data_dict, key)
            losses_one_dataset_recorder, predictions_nps, label_nps, feature_nps = self.test_one_dataset(test_data_loaders[key],epoch)
            losses_all_datasets[key] = losses_one_dataset_recorder
            metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps, img_names=data_dict['image'])
            for metric_name, value in metric_one_dataset.items():
                if metric_name in avg_metric:
                    avg_metric[metric_name] += value
            avg_metric['dataset_dict'][key] = metric_one_dataset[self.metric_scoring]
            if type(self.model) is AveragedModel:
                metric_str = f"Iter Final for SWA:    "
                for k, v in metric_one_dataset.items():
                    metric_str += f"testing-metric, {k}: {v}    "
                self.logger.info(metric_str)
                continue
            self.save_best(epoch, iteration, step, losses_one_dataset_recorder, key, metric_one_dataset)

            # 为每个数据集生成t-SNE图
            tsne_save_dir = os.path.join(self.log_dir, 'test', key, 'tsne')
            os.makedirs(tsne_save_dir, exist_ok=True)
            tsne_save_path = os.path.join(tsne_save_dir, f'{key}_epoch{epoch}_iter{iteration}.png')
            self.visualize_tsne(epoch, iteration, key, feature_nps, label_nps, predictions_nps, tsne_save_path)

        if len(keys) > 0 and self.config.get('save_avg', False):
            for key in avg_metric:
                if key != 'dataset_dict':
                    avg_metric[key] /= len(keys)
            self.save_best(epoch, iteration, step, None, 'avg', avg_metric)

        self.logger.info('===> Test Done!')
        return self.best_metrics_all_time

    @torch.no_grad()
    def inference(self, data_dict,epoch):
        predictions = self.model(data_dict, epoch,inference=True)
        return predictions