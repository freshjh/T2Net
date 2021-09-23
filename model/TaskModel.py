import torch
from torch.autograd import Variable
import util.task as task
from .base_model import BaseModel
from . import network


class TNetModel(BaseModel):
    def name(self):
        return 'TNet Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.loss_names = ['lab_s', 'lab_t', 'lab_smooth']
        self.visual_names = ['img_s', 'lab_s', 'lab_s_g', 'img_t', 'lab_t', 'lab_t_g']
        self.model_names = ['img2task']

        # define the task network
        self.net_img2task = network.define_G(opt.image_nc, opt.label_nc, opt.ngf, opt.task_layers, opt.norm,
                                           opt.activation, opt.task_model_type, opt.init_type, opt.drop_rate,
                                           False, opt.gpu_ids, opt.U_weight)

        if self.isTrain:
            # define the loss function
            self.l1loss = torch.nn.L1Loss()
            self.l2loss = torch.nn.MSELoss()

            self.optimizer_img2task = torch.optim.Adam(self.net_img2task.parameters(), lr=opt.lr_task, betas=(0.9, 0.999))

            self.optimizers = []
            self.schedulers = []

            self.optimizers.append(self.optimizer_img2task)
            for optimizer in self.optimizers:
                self.schedulers.append(network.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)

    def set_input(self, input):
        self.input = input
        self.img_source = input['img_source']
        self.img_target = input['img_target']
        if self.isTrain:
            self.lab_source = input['lab_source']
            self.lab_target = input['lab_target']

        if len(self.gpu_ids) > 0:
            self.img_source = self.img_source.cuda(self.gpu_ids[0], async=True)
            self.img_target = self.img_target.cuda(self.gpu_ids[0], async=True)
            if self.isTrain:
                self.lab_source = self.lab_source.cuda(self.gpu_ids[0], async=True)
                self.lab_target = self.lab_target.cuda(self.gpu_ids[0], async=True)

    def forward(self):
        self.img_s = Variable(self.img_source)
        self.img_t = Variable(self.img_target)
        self.lab_s = Variable(self.lab_source)
        self.lab_t = Variable(self.lab_target)

    def foreward_G_basic(self, net_G, img_s):


        fake = net_G(img_s)

        size = len(fake)

        f_s = fake[0]
        img_fake = fake[1:]

        img_s_fake = []


        for img_fake_i in img_fake:
            img_s = img_fake_i
            img_s_fake.append(img_s)

        return img_s_fake, f_s, size

    def backward_task(self):

        '源域预测，目标域预测，源域特征，目标域特征，batchsize'
        self.lab_s_g, self.lab_f_s, size = \
            self.foreward_G_basic(self.net_img2task, self.img_s)

        'lab_s is the gt of src domain'
        lab_real = task.scale_pyramid(self.lab_s, size-1)
        task_loss = 0
        'supervised training of src data'
        for (lab_fake_i, lab_real_i) in zip(self.lab_s_g, lab_real):
            task_loss += self.l1loss(lab_fake_i, lab_real_i)

        self.loss_lab_s = task_loss

        total_loss = self.loss_lab_s

        total_loss.backward()

    def optimize_parameters(self, epoch_iter):

        self.forward()
        # task network
        self.optimizer_img2task.zero_grad()
        self.backward_task()
        self.optimizer_img2task.step()

    def validation_target(self):

        # TODO 改成在目标域评测
        lab_real = task.scale_pyramid(self.lab_t, len(self.lab_t_g))
        task_loss = 0
        for (lab_fake_i, lab_real_i) in zip(self.lab_t_g, lab_real):
            task_loss += self.l1loss(lab_fake_i, lab_real_i)

        self.loss_lab_t = task_loss * self.opt.lambda_rec_lab