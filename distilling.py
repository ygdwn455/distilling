import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from net import TeacherNet,StudentNet
import pytorch_lightning.callbacks as callbacks
from dataloader import test_dataloader,val_dataloader,train_dataloader
from sklearn.metrics import accuracy_score as accuracy
import warnings
warnings.filterwarnings("ignore")

class KDMoudle(pl.LightningModule):
    def __init__(self, teacher, student, learning_rate, temperature, alpha):
        super().__init__()

        self.teacher = teacher
        self.teacher.requires_grad_(False)
        self.student = student

        self.learning_rate = learning_rate

        self.temperature = temperature
        self.alpha = alpha

    def forward(self, x):
        student_logits = self.student(x)
        teacher_logits = self.teacher(x)

        return student_logits, teacher_logits

    def training_step(self, batch, batch_index):
        x, y = batch
        student_logits, teacher_logits = self.forward(x)

        # # NOTE:第一组：直接用hard_loss训练student网络
        # loss = F.cross_entropy(student_logits, y)
        #
        # # NOTE:第二组：用soft_loss训练student网络
        # loss = nn.KLDivLoss()(F.log_softmax(student_logits / self.temperature),
        #                       F.softmax(teacher_logits / self.temperature)) * (
        #                self.alpha * self.temperature * self.temperature)

        # NOTE:第三组：用hard_loss+soft_loss训练student网络
        soft_loss = nn.KLDivLoss()(F.log_softmax(student_logits / self.temperature),
                                   F.softmax(teacher_logits / self.temperature)) * (
                            self.alpha * self.temperature * self.temperature)
        hard_loss = F.cross_entropy(student_logits, y) * (1.0 - self.alpha)
        loss = hard_loss + soft_loss

        # WHY:student_logits为什么用log_softmax 而 teacher_logits直接用softmax？

        self.log("student_train_loss", loss)
        return loss

    def validation_step(self, batch, batch_index):
        x, y = batch
        student_logits, teacher_logits = self.forward(x)

        student_loss = F.cross_entropy(student_logits, y)

        student_preds = torch.argmax(student_logits, dim=1)
        student_acc = accuracy(student_preds, y)

        teacher_preds = torch.argmax(teacher_logits, dim=1)
        teacher_acc = accuracy(teacher_preds, y)

        self.log("student_val_loss", student_loss, prog_bar=True)
        self.log("student_val_acc", student_acc, prog_bar=True)
        self.log("teacher_val_acc", teacher_acc, prog_bar=True)

        return student_loss

    def test_step(self, batch, batch_index):
        x, y = batch
        student_logits, teacher_logits = self.forward(x)

        student_preds = torch.argmax(student_logits, dim=1)
        student_acc = accuracy(student_preds, y)

        teacher_preds = torch.argmax(teacher_logits, dim=1)
        teacher_acc = accuracy(teacher_preds, y)

        self.log("student_test_acc", student_acc, prog_bar=True)
        self.log("teacher_test_acc", teacher_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.student.parameters(), lr=self.learning_rate)
        return optimizer

class Arguments:
    pass

# callbacks
def get_callbacks():
    # 监控student_val_loss，不再减小了就停止
    early_stopping = callbacks.early_stopping.EarlyStopping(monitor='student_val_loss',
                                                            min_delta=1e-4, patience=2,
                                                            verbose=False, mode='min')
    # checkpoint
    model_checkpoint = callbacks.ModelCheckpoint(save_weights_only=True)

    # 监控学习率
    lr_monitor = callbacks.LearningRateMonitor(logging_interval='step')

    return [early_stopping, model_checkpoint, lr_monitor]

if __name__ == '__main__':
    # %%
    # NOTE:训练参数
    train_args = Arguments()

    train_args.learning_rate = 1e-3
    train_args.max_epochs = 10
    train_args.temperature = 2
    train_args.alpha = 0.8
    # %%
    # 实例化pl_moudle
    teacher = TeacherNet()
    # 载入权重
    teacher.load_state_dict(torch.load("./mnist_cnn.pt"))
    student = StudentNet()
    kd_moudle = KDMoudle(teacher=teacher,
                         student=student,
                         learning_rate=train_args.learning_rate,
                         temperature=train_args.temperature,
                         alpha=train_args.alpha)

    trainer = pl.Trainer(
        # fast_dev_run=1,  # debug时开启，只跑一个batch的train、val和test
        max_epochs=train_args.max_epochs,
        callbacks=get_callbacks(),

        #progress_bar_refresh_rate=20,
        #flush_logs_every_n_steps=1,
        log_every_n_steps=1)

    # %%
    # training
    trainer.fit(kd_moudle, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    # %%
    # testing
    trainer.test(dataloaders=test_dataloader)

    print('finish!!!')





