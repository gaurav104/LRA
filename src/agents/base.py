"""
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""
import logging
import torch
from torch import nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.autograd import Variable
import higher

class BaseAgent:
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        raise NotImplementedError

    def run(self):
        """
        The main operator
        :return:
        """
        raise NotImplementedError

    def train(self):
        """
        Main training loop
        :return:
        """
        raise NotImplementedError

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        raise NotImplementedError


    def distill_knowledge_meta6(self, epoch, gen_imgs):
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
            # gen_imgs_cat = torch.cat((gen_imgs, gen_imgs_mem),dim=0)
            self.student.train()

            bs_gen_imgs = gen_imgs.size(0)
            bs_size_gen_imgs_mem = gen_imgs_mem.size(0)

        with torch.no_grad():
            outputs_T = self.teacher(gen_imgs).detach()
            outputs_T_replay = self.teacher(gen_imgs_mem).detach()

        self.optimizer_S_inner.zero_grad()

        with higher.innerloop_ctx(self.student, self.optimizer_S_inner) as (fmodel_s, diffopt):

            # inner loss only one iteration

            
            inner_output_S = fmodel_s(gen_imgs[:(self.config.batch_size*7)//8])
            inner_loss = F.l1_loss(inner_output_S, outputs_T[:(self.config.batch_size*7)//8])
            diffopt.step(inner_loss)


            # outer test
            output_S_replay = fmodel_s(gen_imgs_mem)
            outer_loss1 = F.l1_loss(output_S_replay, outputs_T_replay)

            outputs_S = fmodel_s(gen_imgs[(self.config.batch_size*7)//8:])
            
            outer_loss2 = F.l1_loss(outputs_S, outputs_T[(self.config.batch_size*7)//8:])

            outputs_S_gen_imgs_cat= self.student(torch.cat((gen_imgs, gen_imgs_mem)))

            loss_kd = F.l1_loss(outputs_S_gen_imgs_cat, torch.cat((outputs_T, outputs_T_replay)))

            total_loss = outer_loss1 + outer_loss2 + loss_kd
            total_loss.backward()
        
        self.optimizer_S.step()

        return total_loss, outer_loss1, loss_kd


    def distill_knowledge_meta7(self, epoch, gen_imgs):
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
            # gen_imgs_cat = torch.cat((gen_imgs, gen_imgs_mem),dim=0)
            self.student.train()

            bs_gen_imgs = gen_imgs.size(0)
            bs_size_gen_imgs_mem = gen_imgs_mem.size(0)

        with torch.no_grad():
            outputs_T = self.teacher(gen_imgs).detach()
            outputs_T_replay = self.teacher(gen_imgs_mem).detach()

        self.optimizer_S_inner.zero_grad()

        with higher.innerloop_ctx(self.student, self.optimizer_S_inner) as (fmodel_s, diffopt):

            # inner loss only one iteration

            
            inner_output_S = fmodel_s(gen_imgs)
            inner_loss = F.l1_loss(inner_output_S, outputs_T)
            diffopt.step(inner_loss)


            # outer test
            output_S_replay = fmodel_s(gen_imgs_mem)
            outer_loss1 = F.l1_loss(output_S_replay, outputs_T_replay)

            # outputs_S = fmodel_s(gen_imgs[(self.config.batch_size*7)//8:])
            
            # outer_loss2 = F.l1_loss(outputs_S, outputs_T[(self.config.batch_size*7)//8:])

            outputs_S_gen_imgs_cat= self.student(torch.cat((gen_imgs, gen_imgs_mem)))

            loss_kd = F.l1_loss(outputs_S_gen_imgs_cat, torch.cat((outputs_T, outputs_T_replay)))

            total_loss = outer_loss1 + loss_kd
            total_loss.backward()
        
        self.optimizer_S.step()

        return total_loss, outer_loss1, loss_kd

 