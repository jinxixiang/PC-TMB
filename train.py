import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import logging
from light import light, light_init
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data.dataloader import default_collate
# customized libs
from utils import read_yaml, Accuracy_Logger, Lookahead, DistributedSamplerWrapper
from trainer import get_model, get_loss, get_optimizer, get_dataset
import torch_geometric
import warnings

torch.manual_seed(20211210)

warnings.filterwarnings("ignore")

# parse arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--work_dir", type=str, default="./work_dir")
parser.add_argument("--config", type=str, default="./config/config01.yaml")
parser.add_argument("--log_file", type=str, default="log.txt")
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
device = torch.device("cuda", args.local_rank)


def create_logger(log_file):
    if args.local_rank < 0:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.WARNING)
        return logger
    log_format = "%(asctime)s %(levelname)5s %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


class Experiment(object):
    def __init__(self, cfg):
        self.cfg = cfg
        # create ddp model
        self.model = get_model(cfg)
        self.model.cuda(device)
        self.model = DDP(self.model, device_ids=[args.local_rank], find_unused_parameters=True)

        # create scheduler
        self.optimizer_cls, self.scheduler_cls = get_optimizer(cfg)
        self.optimizer = self.optimizer_cls(self.model.parameters(), **cfg.Optimizer.optimizer.params)
        # use look ahead optimizer
        if self.cfg.Optimizer.look_ahead:
            self.optimizer = Lookahead(self.optimizer)

        self.scheduler = self.scheduler_cls(self.optimizer, **cfg.Optimizer.lr_scheduler.params)
        self.criterion = get_loss(cfg)

        # whether or not to resume from the latest checkpoint
        resume_path = os.path.exists(os.path.join(args.work_dir, 'latest.pth.tar'))
        self.st_fold = 0
        self.st_epoch = 0
        if resume_path:  # otherwise load from the latest checkpoint
            args.load_weight = os.path.join(args.work_dir, 'latest.pth.tar')

            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(args.load_weight, map_location='cpu')
            self.st_fold = checkpoint['kfold']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['opt'])
            self.scheduler.load_state_dict(checkpoint['sch'])
            self.st_epoch = self.scheduler.last_epoch
            print(f"=> loading the latest checkpoint. kfold={self.st_fold}, epoch={self.scheduler.last_epoch}")

    def refresh(self):
        # refresh model paras after one fold training
        self.model = get_model(self.cfg)
        self.model.cuda(device)
        # self.model = DDP(self.model, device_ids=[args.local_rank], find_unused_parameters=True)
        self.optimizer_cls, self.scheduler_cls = get_optimizer(self.cfg)
        self.optimizer = self.optimizer_cls(self.model.parameters(), **self.cfg.Optimizer.optimizer.params)
        # use look ahead optimizer
        if self.cfg.Optimizer.look_ahead:
            self.optimizer = Lookahead(self.optimizer)

        self.scheduler = self.scheduler_cls(self.optimizer, **self.cfg.Optimizer.lr_scheduler.params)
        self.st_epoch = 0
        torch.cuda.set_device(args.local_rank)

    def run_one_fold(self, kfold, logger):
        # create loader
        train_dataset = get_dataset(self.cfg, kfold, mode='train')
        valid_dataset = get_dataset(self.cfg, kfold, mode='val')

        def collate_MIL_graph(batch):
            elem = batch[0]
            transposed = zip(*batch)
            return [samples[0] if isinstance(samples[0], torch_geometric.data.Batch) else default_collate(samples) for
                    samples in transposed]

        train_sampler = DistributedSampler(train_dataset,
                                           num_replicas=torch.cuda.device_count(),
                                           rank=args.local_rank, shuffle=True)

        # train_sampler = DistributedSamplerWrapper(ImbalancedDatasetSampler(train_dataset),
        #                                           num_replicas=torch.cuda.device_count(),
        #                                           rank=args.local_rank,
        #                                           shuffle=False)

        val_sampler = DistributedSampler(valid_dataset,
                                         num_replicas=torch.cuda.device_count(),
                                         rank=args.local_rank)

        bs = self.cfg.Data.dataloader.batch_size
        num_workers = self.cfg.Data.dataloader.num_workers

        train_loader = DataLoader(train_dataset, batch_size=bs, collate_fn=collate_MIL_graph,
                                  shuffle=False, num_workers=num_workers,
                                  pin_memory=True, drop_last=False, sampler=train_sampler)

        valid_loader = DataLoader(valid_dataset, batch_size=bs, collate_fn=collate_MIL_graph,
                                  shuffle=False, num_workers=num_workers,
                                  pin_memory=True, drop_last=False)

        if args.local_rank == 0:
            logger.info(f"train loader: {len(train_loader)}")
            logger.info(f"val loader: {len(valid_loader)}")

        best_preds, valid_labels = self.train_loop(kfold, train_sampler,
                                                   train_loader, valid_loader, logger)

        return best_preds, valid_labels

    def train_loop(self, kfold, train_sampler, train_loader, valid_loader, logger):

        best_score = -100
        best_preds = []
        valid_labels = []
        avg_loss = 0.0

        # training loop for classification
        for epoch in range(self.st_epoch, self.cfg.General.epochs):
            train_sampler.set_epoch(epoch)
            avg_loss = self.train_one_epoch(train_loader)

            if args.local_rank == 0:
                # save the lastest checkpoint
                torch.save({
                    'kfold': kfold,
                    'state_dict': self.model.state_dict(),
                    'opt': self.optimizer.state_dict(),
                    'sch': self.scheduler.state_dict()
                }, '{}/latest.pth.tar'.format(args.work_dir))

            # evaluate on one gpu
            if (epoch + 1) % self.cfg.General.eval_interval == 0:
                score = self.evaluate_auc(valid_loader, logger)
                # save the best model
                if args.local_rank == 0:

                    if best_score < score:
                        best_score = score

                        logger.info(f"saving best auc: {best_score}")
                        torch.save(self.model.state_dict(), os.path.join(args.work_dir, f'fold{kfold}.pth'))

                    logger.info(f'Fold: {kfold} Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}   AUC: {score}')

            # adjust learning rate
            self.scheduler.step()

        self.st_epoch = 0
        return best_preds, valid_labels

    def train_one_epoch(self, loader):
        if args.local_rank == 0:
            print("training..")
        avg_loss = 0.
        self.model.train()
        for idx, batch in tqdm(enumerate(loader)):

            feat, type, label = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            logits, _, _, result_dict = self.model(feat, type, label, instance_eval=True)
            bag_loss = self.criterion(logits, label)
            inst_loss = result_dict["instance_loss"]
            loss = bag_loss + 0.3 * inst_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            avg_loss += loss.item() / len(loader)

        return avg_loss

    def evaluate_auc(self, loader, logger):
        self.model.eval()
        if args.local_rank == 0:
            print("evaluating...")

        n_classes = 2

        val_loss = 0.
        val_error = 0.
        acc_logger = Accuracy_Logger(n_classes=n_classes)

        prob = np.zeros((len(loader), n_classes))
        labels = np.zeros(len(loader))

        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(loader)):

                feat, type, label = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                instance_eval = False
                logits, Y_prob, Y_hat, results_dict = self.model(feat, type, label, instance_eval)
                loss = self.criterion(logits, label)

                acc_logger.log(Y_hat, label)
                val_loss += loss.item()

                prob[batch_idx] = Y_prob.cpu().numpy()
                labels[batch_idx] = label.item()
                error = 1. - Y_hat.float().eq(label.float()).float().mean().item()
                val_error += error

        val_error /= len(loader)
        val_loss /= len(loader)

        if n_classes == 2:
            auc = roc_auc_score(labels, prob[:, 1])
        else:
            auc = roc_auc_score(labels, prob, multi_class='ovr')

        if args.local_rank == 0:

            for i in range(n_classes):
                acc, correct, count = acc_logger.get_summary(i)
                logger.info('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

            logger.info('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
            print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))

        return auc


params = {
    "training_framework": "pytorch_ddp",
    "enable_optimizations": True
}


@light_init(params)
def main():
    # create work_dir
    os.makedirs(args.work_dir, exist_ok=True)

    # create log file
    log_file = os.path.join(args.work_dir, args.log_file)
    logger = create_logger(log_file)

    # read config
    cfg = read_yaml(args.config)
    if args.local_rank == 0:
        for key, value in cfg.items():
            logger.info(f"{key.ljust(30)}: {value}")

    # train start
    preds_final = []
    labels_final = []

    # init an object
    expt = Experiment(cfg)
    st_fold = expt.st_fold

    # run one fold for fixed train-test-val split
    if not cfg.General.cross_validation:
        cfg.General.num_folds = 1

    for kfold in range(st_fold, cfg.General.num_folds):
        pred, label = expt.run_one_fold(kfold, logger)
        preds_final.append(pred)
        labels_final.append(label)
        # refresh params for a new fold training
        expt.refresh()


if __name__ == "__main__":
    main()
