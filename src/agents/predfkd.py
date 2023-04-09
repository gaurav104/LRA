"""
Main Agent for predfkd
"""
import os
import random
import numpy as np
import time
import shutil

import torch
from torch import nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.autograd import Variable

from agents.base import BaseAgent
from graphs.models.networks.generator import Generator, Encoder, MemoryGenerator
from graphs.models.model_utils import init_model
from graphs.models.resnet import ResNet34
from graphs.models.myresnet import MyResNet34
from graphs.models.networks.lenet import LeNet5

# from graphs.models.condensenet import CondenseNet
from graphs.losses.kd_losses import JS_divergence

from datasets.dfkd import DFKDDataLoader

from tensorboardX import SummaryWriter
from utils.constants import num_classes, num_channels
from utils.metrics import AverageMeter
from utils.misc import print_cuda_statistics
from utils.train_utils import adjust_learning_rate

# cudnn.benchmark = True
# cudnn.enabled = True

class PREDFKDAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        # initialize my counters
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_acc = 0
        self.current_accuracy = 0
        # Check is cuda is available or not
        self.is_cuda = torch.cuda.is_available()
        # Construct the flag and make sure that cuda is available
        self.cuda = self.is_cuda & self.config.cuda

        if self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(self.config.seed)
            # torch.cuda.set_device(self.config.gpu_device)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            self.logger.info("Operation will be on *****GPU-CUDA***** ")
            # print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.config.seed)
            self.logger.info("Operation will be on *****CPU***** ")

        np.random.seed(self.config.seed)
        random.seed(self.config.seed)


        # Initialize generators 
        self.novel_generator = Generator(self.config, num_channels).to(self.device)
        self.mem_generator = MemoryGenerator(self.config, num_channels).to(self.device)
        self.encoder = Encoder(self.config, num_channels).to(self.device)

        if torch.cuda.device_count() > 1:
            self.novel_generator = nn.DataParallel(self.novel_generator)
            self.mem_generator = nn.DataParallel(self.mem_generator)
            self.encoder = nn.DataParallel(self.encoder)

        self.optimizer_G = torch.optim.Adam(self.novel_generator.parameters(), lr=self.config.lr_G )
        self.optimizer_G2 = torch.optim.Adam(self.mem_generator.parameters(), lr=self.config.lr_G / 40 )
        self.optimizer_E = torch.optim.Adam(self.encoder.parameters(), lr=self.config.lr_G / 40 )

        #Initialize student
        self.student = init_model(self.config)
        if torch.cuda.device_count() > 1:
            self.student = nn.DataParallel(self.student)

        #Initilize test dataloader
        self.data_loader = DFKDDataLoader(self.config) # 
        self.data_loader.load_val_dataset() # attributes :- data_test, data_test_loader

        #Initilizing Average Meter
        self.avg_meter = AverageMeter()


        if self.config.dataset == 'MNIST':
            self.optimizer_S = torch.optim.SGD(self.student.parameters(), lr=self.config.lr_S, momentum=0.9, weight_decay=5e-4)
            self.scheduler_S = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_S, T_max=self.config.max_epoch)
            # self.optimizer_S = torch.optim.Adam(self.student.parameters(), lr=self.config.lr_S)
            # self.scheduler_S = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_S, T_max=self.config.max_epoch)

        else:
            self.optimizer_S = torch.optim.SGD(self.student.parameters(), lr=self.config.lr_S, momentum=0.9, weight_decay=5e-4)   
        self.scheduler_S = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_S, T_max=self.config.max_epoch)

        
        # self.optimizer_S = torch.optim.SGD(self.student.parameters(), lr=self.config.lr_S, momentum=0.9, weight_decay=5e-4)   
        # self.scheduler_S = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_S, T_max=self.config.max_epoch)

        if self.config.mode=='train':
            if self.config.dataset =='cifar100':

                self.teacher = torch.load(self.config.teacher_dir + self.config.teacher_name).to(self.device)
                self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

            if self.config.dataset == 'MNIST':
                self.teacher = LeNet5()
                # checkpoint = torch.load(os.path.join(self.config.teacher_dir, self.config.teacher_name))
                # self.teacher.load_state_dict(checkpoint['state_dict'])
                
                self.teacher.load_state_dict(torch.load(os.path.join(self.config.teacher_dir,self.config.teacher_name)))
                self.teacher.to(self.device)


                print("Teacher Model loaded successfully")
                self.criterion = torch.nn.CrossEntropyLoss().to(self.device)



            if self.config.dataset == 'cifar10':
                self.teacher = ResNet34(self.config).to(self.device)
                checkpoint = torch.load(os.path.join(self.config.teacher_dir, self.config.teacher_name))
                self.teacher.load_state_dict(checkpoint['state_dict'])
                print("Teacher Model loaded successfully")
                # self.model.load_state_dict(checkpoint['state_dict'])
                # self.teacher.to(self.device)
                self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

            if self.config.dataset == 'SVHN':
                self.teacher = ResNet34(self.config).to(self.device)
                checkpoint = torch.load(os.path.join(self.config.teacher_dir, self.config.teacher_name))
                self.teacher.load_state_dict(checkpoint['state_dict'])
                print("Teacher Model loaded successfully")
                # self.model.load_state_dict(checkpoint['state_dict'])
                # self.teacher.to(self.device)
                self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

            if self.config.dataset == 'FMNIST':
                self.teacher = ResNet34(self.config).to(self.device)
                checkpoint = torch.load(os.path.join(self.config.teacher_dir, self.config.teacher_name))
                self.teacher.load_state_dict(checkpoint['state_dict'])
                print("Teacher Model loaded successfully")
                # self.model.load_state_dict(checkpoint['state_dict'])
                # self.teacher.to(self.device)
                self.criterion = torch.nn.CrossEntropyLoss().to(self.device)


            if self.config.dataset == 'tiny-imagenet':
                # self.teacher = torch.load(self.config.teacher_dir + self.config.teacher_name).to(self.device)
                # self.criterion = torch.nn.CrossEntropyLoss().to(self.device)


                self.teacher = MyResNet34().to(self.device)
                checkpoint = torch.load(os.path.join(self.config.teacher_dir, self.config.teacher_name))
                self.teacher.load_state_dict(checkpoint['state_dict'])
                print("Teacher Model loaded successfully")
                # self.model.load_state_dict(checkpoint['state_dict'])
                # self.teacher.to(self.device)
                self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

                # test_input = torch.randn((5, 3, 64, 64),device=self.device)

                # out =  self.teacher(test_input)
                # sys.exit()



            if torch.cuda.device_count() > 1:
                self.teacher = nn.DataParallel(self.teacher)
                
            self.teacher.eval()



        # -------------
        #  Distillation
        # -------------
        print('DISTILLATION STARTED')
        self.accr = 0
        self.config.num_novel_samples = self.config.batch_size
        self.config.num_mem_samples = (self.config.batch_size // 8)
        self.config.mem_gen_upd_period = 1
        self.config.student_rehearse_period = 1

        # self.model = self.model.to(self.device)
        # self.loss = self.loss.to(self.device)
        # Model Loading from the latest checkpoint if not found start from scratch.
        # self.load_checkpoint(self.config.checkpoint_file)
        # Tensorboard Writer
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment=self.config.exp_name)

    def save_checkpoint(self, filename='checkpoint.pth.tar', is_best=0):
        """
        Saving the latest checkpoint of the training
        :param filename: filename which will contain the state
        :param is_best: flag is it is the best model
        :return:
        """
        state = {
            'epoch': self.current_epoch,
            'student': self.student.state_dict(),
            'encoder': self.encoder.state_dict(),
            'mem_generator': self.mem_generator.state_dict(),
            'novel_generator': self.novel_generator.state_dict(),
            'optimizer_S': self.optimizer_S.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_G2': self.optimizer_G2.state_dict(),
            'optimizer_E': self.optimizer_E.state_dict(),
            'scheduler_S': self.scheduler_S.state_dict(),
            'avg_meter': self.avg_meter,
            'current_accuracy': self.current_accuracy
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + filename)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + filename,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def load_checkpoint(self, filename):
        filename = self.config.checkpoint_dir + filename
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            epoch = checkpoint['epoch']
            self.logger.info("Model found loaded of epoch {}, Training will start from epoch {}".format(epoch,epoch + 1))

            self.current_epoch = epoch + 1
            self.student.load_state_dict(checkpoint['student'])
            self.encoder.load_state_dict(checkpoint['encoder'])
            self.mem_generator.load_state_dict(checkpoint['mem_generator'])
            self.novel_generator.load_state_dict(checkpoint['novel_generator'])
            self.optimizer_S.load_state_dict(checkpoint['optimizer_S'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            self.optimizer_G2.load_state_dict(checkpoint['optimizer_G2'])
            self.optimizer_E.load_state_dict(checkpoint['optimizer_E'])
            self.scheduler_S.load_state_dict(checkpoint['scheduler_S'])
            self.avg_meter = checkpoint['self.avg_meter']
            self.current_accuracy = checkpoint['current_accuracy']

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                             .format(self.config.checkpoint_dir, self.current_epoch))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def run(self):
        """
        This function will the operator
        :return:
        """
        try:
            if self.config.mode == 'test':
                self.validate()
            else:

                self.start_time = time.time()
                self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")
            self.finalize()

    def train(self):
        """
        Main training function, with per-epoch model saving
        """
        # avg_accuracy = AverageMeter()

        start_epoch = self.current_epoch

        for epoch in range(start_epoch, self.config.max_epoch):

            for i in range(int(36 *(1024//self.config.batch_size))):

                # Train novel generator
                for k in range(int(self.config.gen_iter)):
                    novel_gen_losses = self.train_novel_generator()

                # Distill Knowledge
                for dist_step in range(int(self.config.kd_iter)):
                    # Generate new synthetic samples
                    z = Variable(torch.randn(self.config.num_novel_samples, self.config.latent_dim), requires_grad=False).to(self.device)
                    with torch.no_grad():
                        gen_imgs = self.novel_generator(z)
                    loss_kd = self.distill_knowledge(epoch, gen_imgs)

                    # Memory Generator Training
                    if (epoch % self.config.mem_gen_upd_period == 0) & (dist_step < (int(4))):
                        mem_gen_loss = self.train_memory_generator(epoch, i, gen_imgs)

                if i == 1:
                    print("[Epoch %d/%d] [loss_G: %f] [loss_G2: %f] [loss_S: %f]" % (epoch, self.config.max_epoch, mem_gen_loss.item(), novel_gen_losses[0].item(), loss_kd.item()))
                    print('Time elapsed: %.2f seconds' % (time.time() - self.start_time))


            self.current_epoch = epoch

            # if self.config.dataset != "MNIST":

            self.scheduler_S.step()

            self.current_accuracy = self.eval_model()

            self.avg_meter.update(self.current_accuracy)

            self.summary_writer.add_scalar('Epoch/Test_Accuracy', self.current_accuracy, self.current_epoch)
            self.summary_writer.add_scalar('Epoch/Cumulative_Accuracy', self.avg_meter.val, self.current_epoch)
            self.save_checkpoint(self.config.checkpoint_filename)


    def train_memory_generator(self, epoch, i, gen_imgs):
        self.mem_generator.train()
        self.encoder.train()
        self.optimizer_G2.zero_grad()
        self.optimizer_E.zero_grad()
        
        # If memory generator has not seen any samples just use novel samples
        if (i == 0) & (epoch == 0):
            with torch.no_grad():
                outputs_T = self.teacher(gen_imgs)
            z_enc, mu, log_var = self.encoder(gen_imgs[:(self.config.batch_size // 2)].detach())
            gen_imgs_rec = self.mem_generator(z_enc)
            out_T_target = [gen_imgs[:(self.config.batch_size // 2)].detach()]
            out_T_target.extend([outputs_T[:(self.config.batch_size // 2)]])
            #out_T_target.extend(features_T[-1:])

            rec_pred_T = self.teacher(gen_imgs_rec)
            rec_out_T = [gen_imgs_rec]
            rec_out_T.extend([rec_pred_T])
            #rec_out_T.extend(rec_features_T[-1:])
            
            # Reconstruction Loss
            recons_loss = 0.
            for rec_idx, rec_target in enumerate(out_T_target):
                recons_loss += F.l1_loss(rec_out_T[rec_idx], rec_target[:(self.config.batch_size // 2)].detach())

        else:
            z_mem = torch.randn((self.config.num_mem_samples, self.config.latent_dim), requires_grad=True, device='cuda')
            optimizer_Z = torch.optim.Adam([z_mem], lr=self.config.lr_G)
            optimizer_Z.zero_grad()
            
            # Tune latent variables
            for z_epoch in range(1):
                log_var = torch.var(z_mem, dim=0)
                mu = torch.mean(z_mem, dim=0)
                gen_imgs_mem = self.mem_generator(z_mem)
                outputs_T = self.teacher(gen_imgs_mem)
                softmax_o_T = torch.nn.functional.softmax(outputs_T, dim = 1).mean(dim = 0)
                loss_one_hot = torch.nn.CrossEntropyLoss()(outputs_T,outputs_T.data.max(1)[1])
                loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()
                loss_kld = F.mse_loss(log_var, torch.ones(self.config.latent_dim).cuda()) + F.mse_loss(mu, torch.zeros(self.config.latent_dim).cuda())

                loss_match = loss_information_entropy*self.config.ie + loss_one_hot*(self.config.oh)
                (loss_match + loss_kld).backward()
                optimizer_Z.step()

            # Self infer memory samples
            with torch.no_grad():
                gen_imgs_mem = self.mem_generator(z_mem).detach()
                concat_imgs = torch.cat((gen_imgs[:(self.config.batch_size // 2)].detach(), gen_imgs_mem),dim=0)
                pred_T_target = self.teacher(concat_imgs)
                out_T_target = [concat_imgs]
                out_T_target.extend([pred_T_target])
            #out_T_target.extend(features_T[-1:])
            
            z_enc, mu, log_var = self.encoder(concat_imgs)
            gen_imgs_rec = self.mem_generator(z_enc)
            rec_pred_T = self.teacher(gen_imgs_rec)
            rec_out_T = [gen_imgs_rec]
            rec_out_T.extend([rec_pred_T])
            #rec_out_T.extend(rec_features_T[-1:])

            # Reconstruction Loss
            recons_loss = 0.
            for rec_idx, rec_target in enumerate(out_T_target):
                recons_loss += F.l1_loss(rec_out_T[rec_idx], rec_target.detach())

        # KLD Loss
        kld_loss = 1e-5 * torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        mem_gen_loss = recons_loss + kld_loss
        
        mem_gen_loss.backward()

        self.optimizer_G2.step()
        self.optimizer_E.step()

        return mem_gen_loss



    def distill_knowledge(self, epoch, gen_imgs):
        self.optimizer_S.zero_grad()

        # Infer memory Samples
        if ((epoch % self.config.student_rehearse_period) == 0):
            self.mem_generator.eval()
            self.student.eval()
            z_mem = torch.randn((self.config.num_mem_samples, self.config.latent_dim), requires_grad=True, device=self.device)
            optimizer_Z = torch.optim.Adam([z_mem], lr=self.config.lr_G)
            optimizer_Z.zero_grad()
            for z_epoch in range(1):
                log_var = torch.var(z_mem, dim=0)
                mu = torch.mean(z_mem, dim=0)
                gen_imgs_mem = self.mem_generator(z_mem)
                outputs_T = self.teacher(gen_imgs_mem)
                softmax_o_T = torch.nn.functional.softmax(outputs_T, dim = 1).mean(dim = 0)
                loss_one_hot = torch.nn.CrossEntropyLoss()(outputs_T,outputs_T.data.max(1)[1])
                loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()

                loss_kld = F.mse_loss(log_var, torch.ones(self.config.latent_dim).to(self.device)) + F.mse_loss(mu, torch.zeros(self.config.latent_dim).to(self.device))
                loss_match = loss_information_entropy*self.config.ie + loss_one_hot*(self.config.oh)
                (loss_match + loss_kld).backward()
                optimizer_Z.step()

            with torch.no_grad():
                gen_imgs_mem = self.mem_generator(z_mem).detach()
                gen_imgs = torch.cat((gen_imgs, gen_imgs_mem),dim=0)
            
        self.student.train()

        outputs_S = self.student(gen_imgs)

        with torch.no_grad():
            outputs_T = self.teacher(gen_imgs)
        
        loss_kd = F.l1_loss(outputs_S, outputs_T.detach())
        loss_kd.backward()
        self.optimizer_S.step()

        return loss_kd



    def train_novel_generator(self):

        self.novel_generator.train()
        self.optimizer_G.zero_grad()

        # Sample novel synthetic data
        z = Variable(torch.randn(self.config.batch_size, self.config.latent_dim), requires_grad=False).to(self.device)
        gen_imgs = self.novel_generator(z)
         
        # Get teacher responses to match training data dist. 
        outputs_T, features_T = self.teacher(gen_imgs, out_feature=True)     
        pred = outputs_T.data.max(1)[1]
        loss_activation = -features_T[-1].abs().mean()
        loss_one_hot = torch.nn.CrossEntropyLoss()(outputs_T,pred)
        softmax_o_T = torch.nn.functional.softmax(outputs_T, dim = 1).mean(dim = 0)
        loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()
        loss_match_T = loss_one_hot * self.config.oh + loss_information_entropy * self.config.ie + loss_activation * self.config.a

        # Calculate T-S disagreement
        loss_disaggreement = JS_divergence(self.student(gen_imgs), outputs_T.detach())

        loss_gen = loss_match_T + loss_disaggreement
        losses = [loss_gen, loss_disaggreement]

        loss_gen.backward()
        self.optimizer_G.step()

        return losses


    def eval_model(self):

        data_test = self.data_loader.data_test
        data_test_loader = self.data_loader.data_test_loader

        self.student.eval()
        total_correct = 0
        avg_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                output = self.student(images)
                avg_loss += torch.nn.CrossEntropyLoss()(output, labels).sum()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()

        avg_loss /= len(data_test)
        print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), float(total_correct) / len(data_test)))
        accr = round(float(total_correct) / len(data_test), 4)
        return accr


    # def train_one_epoch(self):
    #     """
    #     One epoch training function
    #     """
    #     # Initialize tqdm
    #     tqdm_batch = tqdm(self.data_loader.train_loader, total=self.data_loader.train_iterations,
    #                       desc="Epoch-{}-".format(self.current_epoch))
    #     # Set the model to be in training mode
    #     self.model.train()
    #     # Initialize your average meters
    #     epoch_loss = AverageMeter()
    #     top1_acc = AverageMeter()
    #     top5_acc = AverageMeter()

    #     current_batch = 0
    #     for x, y in tqdm_batch:
    #         if self.cuda:
    #             x, y = x.cuda(async=self.config.async_loading), y.cuda(async=self.config.async_loading)

    #         # current iteration over total iterations
    #         progress = float(self.current_epoch * self.data_loader.train_iterations + current_batch) / (
    #                 self.config.max_epoch * self.data_loader.train_iterations)
    #         # progress = float(self.current_iteration) / (self.config.max_epoch * self.data_loader.train_iterations)
    #         x, y = Variable(x), Variable(y)
    #         lr = adjust_learning_rate(self.optimizer, self.current_epoch, self.config, batch=current_batch,
    #                                   nBatch=self.data_loader.train_iterations)
    #         # model
    #         pred = self.model(x, progress)
    #         # loss
    #         cur_loss = self.loss(pred, y)
    #         if np.isnan(float(cur_loss.item())):
    #             raise ValueError('Loss is nan during training...')
    #         # optimizer
    #         self.optimizer.zero_grad()
    #         cur_loss.backward()
    #         self.optimizer.step()

    #         top1, top5 = cls_accuracy(pred.data, y.data, topk=(1, 5))

    #         epoch_loss.update(cur_loss.item())
    #         top1_acc.update(top1.item(), x.size(0))
    #         top5_acc.update(top5.item(), x.size(0))

    #         self.current_iteration += 1
    #         current_batch += 1

    #         self.summary_writer.add_scalar("epoch/loss", epoch_loss.val, self.current_iteration)
    #         self.summary_writer.add_scalar("epoch/accuracy", top1_acc.val, self.current_iteration)
    #     tqdm_batch.close()

    #     self.logger.info("Training at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
    #         epoch_loss.val) + "- Top1 Acc: " + str(top1_acc.val) + "- Top5 Acc: " + str(top5_acc.val))

    # def validate(self):
    #     """
    #     One epoch validation
    #     :return:
    #     """
    #     tqdm_batch = tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations,
    #                       desc="Valiation at -{}-".format(self.current_epoch))

    #     # set the model in training mode
    #     self.model.eval()

    #     epoch_loss = AverageMeter()
    #     top1_acc = AverageMeter()
    #     top5_acc = AverageMeter()

    #     for x, y in tqdm_batch:
    #         if self.cuda:
    #             x, y = x.cuda(async=self.config.async_loading), y.cuda(async=self.config.async_loading)

    #         x, y = Variable(x), Variable(y)
    #         # model
    #         pred = self.model(x)
    #         # loss
    #         cur_loss = self.loss(pred, y)
    #         if np.isnan(float(cur_loss.item())):
    #             raise ValueError('Loss is nan during validation...')

    #         top1, top5 = cls_accuracy(pred.data, y.data, topk=(1, 5))
    #         epoch_loss.update(cur_loss.item())
    #         top1_acc.update(top1.item(), x.size(0))
    #         top5_acc.update(top5.item(), x.size(0))

    #     self.logger.info("Validation results at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
    #         epoch_loss.avg) + "- Top1 Acc: " + str(top1_acc.val) + "- Top5 Acc: " + str(top5_acc.val))

    #     tqdm_batch.close()

    #     return top1_acc.avg

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        # self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        # self.summary_writer.close()
        # self.data_loader.finalize()