"""OCITGAN module."""

import warnings

import numpy as np
import pandas as pd
import torch
from packaging import version
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional

from data_sampler import DataSampler
from data_transformer import DataTransformer
from base import BaseSynthesizer, random_state
from rdt.transformers import OneHotEncoder
import copy
from collections import Counter
import random
import os





def seed_torch(seed=420):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


class Discriminator(Module):
    def __init__(self, input_dim, discriminator_dim, pac=1):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=1, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))


class Residual(Module):
    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class BackboneNet(Module):
    """BackboneNet for the Generator."""
    def __init__(self, embedding_dim, backbone_dim, data_dim):
        super(BackboneNet, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(backbone_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq += [Linear(dim, data_dim), ReLU()]
        self.seq = Sequential(*seq)

    def forward(self, input_):
        return self.seq(input_)


class BranchNet(Module):
    """BranchNet for the Generator."""
    def __init__(self, data_dim, branch_dim):
        super(BranchNet, self).__init__()
        dim = data_dim
        seq = []
        if branch_dim != ():
            for item in list(branch_dim):
                seq += [Linear(dim, item), LeakyReLU(0.2)]
                dim = item
        seq += [Linear(dim, data_dim)]
        self.seq = Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


class Classifier(Module):
    def __init__(self, input_dim, classifier_dim, num_classes):
        super(Classifier, self).__init__()
        dim = input_dim
        seq = []
        for item in list(classifier_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2)]
            dim = item

        seq += [Linear(dim, num_classes)]
        self.seq = Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


class OCITGAN(BaseSynthesizer):
    def __init__(self, embedding_dim=100,
                 backbone_dim=(50, 50),branch_dim=(50,50),
                 discriminator_dim=(50, 50),classifier_dim = (100,50),
                 backbone_lr=2e-4, backbone_decay=1e-6,
                 branch_lr=2e-4, branch_decay=1e-6,
                 discriminator_lr=2e-4, discriminator_decay=1e-6,
                 classifier_lr=2e-4, classifier_decay=1e-6,
                 batch_size=100, discriminator_steps=1, generator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=1, lambda_=10,
                 alpha=0.5, beta=0.2, gamma=0.2, cuda=True):

        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._backbone_dim = backbone_dim
        self._branch_dim = branch_dim
        self._discriminator_dim = discriminator_dim
        self._classifier_dim = classifier_dim

        self._backbone_lr = backbone_lr
        self._backbone_decay = backbone_decay
        self._branch_lr = branch_lr
        self._branch_decay = branch_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay
        self._classifier_lr = classifier_lr
        self._classifier_decay = classifier_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._generator_steps = generator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac
        self.lambda_ = lambda_
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._backbone = None
        self._discriminator = None
        self._classifier = None

        self._train_data = None
        self._target = None
        self._target_one_hot = None
        self._target_value_list = None
        self._num_classes = None
        self._n_categories = None

        self._each_class_train_data_dict = {}
        self._each_class_data_sampler_dict = {}
        self._target_one_hot_dict = {}
        self._branch_dict = {}
        self._optimizerBranch_dict = {}
        self._each_class_centriod_dict = {}

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.
        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.
        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        if version.parse(torch.__version__) < version.parse('1.2.0'):
            for i in range(10):
                transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard,
                                                        eps=eps, dim=dim)
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError('gumbel_softmax returning NaN.')

        return functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.
        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')

    def _get_target_one_hot(self, target):
        ohe = OneHotEncoder()
        ohe.fit(pd.DataFrame(target, columns=['target']), 'target')
        target_one_hot = ohe.transform(pd.DataFrame(target, columns=['target'])).to_numpy()
        target_one_hot = np.float32(target_one_hot)
        return target_one_hot

    def _is_discrete_column(self, column_info):
        return (len(column_info) == 1 and column_info[0].activation_fn == 'softmax')

    def _before_sampling(self, train_data, data_dim):
        for target_value in self._target_value_list:
            each_class_train_data = train_data[np.array(self._target) == target_value]
            self._each_class_train_data_dict[target_value] = each_class_train_data
            self._each_class_centriod_dict[target_value] = each_class_train_data.mean(0)
            self._each_class_data_sampler_dict[target_value] = DataSampler(
                each_class_train_data,
                self._transformer.output_info_list,
                self._log_frequency)
            self._target_one_hot_dict[target_value] = self._target_one_hot[np.array(self._target) == target_value]
            self._branch_dict[target_value] = BranchNet(data_dim, self._branch_dim).to(self._device)
            self._optimizerBranch_dict[target_value] = optim.Adam(
                self._branch_dict[target_value].parameters(),
                lr=self._branch_lr,
                betas=(0.5, 0.9),
                weight_decay=self._branch_decay)

    @random_state
    def fit(self, train_data, target, discrete_columns=(), epochs=None):
        self._validate_discrete_columns(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated and will be removed '
                 'in a future version. Please pass `epochs` to the constructor instead'),
                DeprecationWarning
            )

        self._train_data = train_data
        self._target = target

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)
        train_data = self._transformer.transform(train_data)
        data_dim = self._transformer.output_dimensions

        target_one_hot = self._get_target_one_hot(target)
        self._target_one_hot = target_one_hot

        num_classes = len(set(target))
        self._num_classes = num_classes

        self._n_categories = sum([
            column_info[0].dim
            for column_info in self._transformer.output_info_list
            if self._is_discrete_column(column_info)])

        self._target_value_list = sorted(list(set(np.array(target))))

        self._backbone = BackboneNet(
            self._embedding_dim + self._n_categories + num_classes,
            self._backbone_dim,
            data_dim
        ).to(self._device)

        optimizerBackbone = optim.Adam(
            self._backbone.parameters(), lr=self._backbone_lr, betas=(0.5, 0.9),
            weight_decay=self._backbone_decay
        )

        self._discriminator = Discriminator(
            data_dim + self._n_categories + num_classes,
            self._discriminator_dim,
            pac=self.pac
        ).to(self._device)

        optimizerD = optim.Adam(
            self._discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )

        self._classifier = Classifier(data_dim, self._classifier_dim, num_classes).to(self._device)
        optimizerC = optim.Adam(
            self._classifier.parameters(), lr=self._classifier_lr,
            betas=(0.5, 0.9), weight_decay=self._classifier_decay
        )
        classifier_loss = torch.nn.CrossEntropyLoss(size_average=True)

        self._before_sampling(train_data, data_dim)

        loss_d_list = []
        loss_pen_list = []
        loss_class_r_list = []
        loss_adv_list = []
        loss_cluster_list = []
        loss_class_g_list = []
        loss_cond_list = []

        '''training'''
        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in range(epochs):
            for id_ in range(steps_per_epoch):
                '''train discriminator'''
                for n_d in range(self._discriminator_steps):
                    num_each_class_dict = {}
                    while True:
                        num_each_class_dict = dict(Counter(np.random.choice(self._target_value_list, self._batch_size)))
                        num_each_class_dict = dict(sorted(num_each_class_dict.items(), key=lambda x: x[0]))
                        if set(num_each_class_dict.keys()) == set(self._target_value_list):
                            break

                    fake_cat_dict = {}
                    real_cat_dict = {}
                    real_dict = {}
                    tar_dict = {}
                    for target_value in self._target_value_list:
                        num_each_class = num_each_class_dict[target_value]
                        mean = torch.zeros(num_each_class, self._embedding_dim, device=self._device)
                        std = mean + 1
                        fakez = torch.normal(mean=mean, std=std)
                        condvec = self._each_class_data_sampler_dict[target_value].sample_condvec(num_each_class)

                        if condvec is None:
                            c1, m1, col, opt = None, None, None, None
                            real, idx = self._each_class_data_sampler_dict[target_value].sample_data(num_each_class,
                                                                                                     col, opt)
                            tar = torch.from_numpy(self._target_one_hot_dict[target_value][idx]).to(self._device)
                            fakez = torch.cat([fakez, tar], dim=1)
                        else:
                            c1, m1, col, opt = condvec
                            c1 = torch.from_numpy(c1).to(self._device)
                            m1 = torch.from_numpy(m1).to(self._device)

                            real, idx = self._each_class_data_sampler_dict[target_value].sample_data(num_each_class,
                                                                                                     col, opt)
                            tar = torch.from_numpy(self._target_one_hot_dict[target_value][idx]).to(self._device)
                            fakez = torch.cat([fakez, c1, tar], dim=1)

                        fake = self._backbone(fakez)
                        fake = self._branch_dict[target_value](fake)
                        fakeact = self._apply_activate(fake)
                        real = torch.from_numpy(real.astype('float32')).to(self._device)
                        real_dict[target_value] = real
                        tar_dict[target_value] = tar

                        if c1 is None:
                            fake_cat = torch.cat([fakeact, tar], dim=1)
                            real_cat = torch.cat([real, tar], dim=1)
                        else:
                            fake_cat = torch.cat([fakeact, c1, tar], dim=1)
                            real_cat = torch.cat([real, c1, tar], dim=1)
                        fake_cat_dict[target_value] = fake_cat
                        real_cat_dict[target_value] = real_cat

                    fake_cat_d = torch.cat(list(fake_cat_dict.values()), dim=0)
                    real_cat_d = torch.cat(list(real_cat_dict.values()), dim=0)
                    real_classifier = torch.cat(list(real_dict.values()), dim=0)
                    tar_classifier = torch.cat(list(tar_dict.values()), dim=0)

                    perm = np.arange(self._batch_size)
                    np.random.shuffle(perm)

                    fake_cat_d = fake_cat_d[perm]
                    real_cat_d = real_cat_d[perm]
                    real_classifier = real_classifier[perm]
                    tar_classifier = tar_classifier[perm]

                    y_fake = self._discriminator(fake_cat_d)
                    y_real = self._discriminator(real_cat_d)

                    pen = self._discriminator.calc_gradient_penalty(
                        real_data=real_cat_d,
                        fake_data=fake_cat_d,
                        device=self._device,
                        pac=self.pac,
                        lambda_=self.lambda_)

                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad()
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                    classifier_output_r = self._classifier(real_classifier)
                    tar_r = torch.argmax(tar_classifier, dim=1)
                    loss_class_r = classifier_loss(classifier_output_r, tar_r)

                    optimizerC.zero_grad()
                    loss_class_r.backward()
                    optimizerC.step()

                '''train generator'''
                for n_g in range(self._generator_steps):
                    num_each_class_dict = {}
                    while True:
                        num_each_class_dict = dict(Counter(np.random.choice(self._target_value_list, self._batch_size)))
                        num_each_class_dict = dict(sorted(num_each_class_dict.items(), key=lambda x: x[0]))
                        if set(num_each_class_dict.keys()) == set(self._target_value_list):
                            break

                    fakeact_dict = {}
                    tar_dict = {}
                    c1_dict = {}
                    m1_dict = {}
                    fake_dict = {}
                    centriod_dict = {}
                    for target_value in self._target_value_list:
                        num_each_class = num_each_class_dict[target_value]
                        mean = torch.zeros(num_each_class, self._embedding_dim, device=self._device)
                        std = mean + 1
                        fakez = torch.normal(mean=mean, std=std)
                        condvec = self._each_class_data_sampler_dict[target_value].sample_condvec(num_each_class)

                        if condvec is None:
                            c1, m1, col, opt = None, None, None, None
                            real, idx = self._each_class_data_sampler_dict[target_value].sample_data(num_each_class,
                                                                                                     col, opt)
                            tar = torch.from_numpy(self._target_one_hot_dict[target_value][idx]).to(self._device)
                            fakez = torch.cat([fakez, tar], dim=1)
                            c1_dict = None
                            m1_dict = None
                        else:
                            c1, m1, col, opt = condvec
                            c1 = torch.from_numpy(c1).to(self._device)
                            m1 = torch.from_numpy(m1).to(self._device)

                            real, idx = self._each_class_data_sampler_dict[target_value].sample_data(num_each_class,
                                                                                                     col, opt)
                            tar = torch.from_numpy(self._target_one_hot_dict[target_value][idx]).to(self._device)
                            fakez = torch.cat([fakez, c1, tar], dim=1)
                            c1_dict[target_value] = c1
                            m1_dict[target_value] = m1

                        fake = self._backbone(fakez)
                        fake = self._branch_dict[target_value](fake)
                        fake_dict[target_value] = fake
                        fakeact = self._apply_activate(fake)
                        fakeact_dict[target_value] = fakeact
                        tar_dict[target_value] = tar
                        centriod = np.tile(self._each_class_centriod_dict[target_value],(num_each_class, 1))
                        centriod_dict[target_value] = torch.from_numpy(centriod.astype('float32')).to(self._device)

                    fake_g = torch.cat(list(fake_dict.values()), dim=0)
                    fakeact_g = torch.cat(list(fakeact_dict.values()), dim=0)
                    tar_g = torch.cat(list(tar_dict.values()), dim=0)
                    centriod_g = torch.cat(list(centriod_dict.values()), dim=0)

                    perm = np.arange(self._batch_size)
                    np.random.shuffle(perm)

                    fake_g = fake_g[perm]
                    fakeact_g = fakeact_g[perm]
                    tar_g = tar_g[perm]
                    centriod_g = centriod_g[perm]

                    if c1_dict is not None:
                        c1_g = torch.cat(list(c1_dict.values()), dim=0)
                        c1_g = c1_g[perm]
                    if m1_dict is not None:
                        m1_g = torch.cat(list(m1_dict.values()), dim=0)
                        m1_g = m1_g[perm]

                    classifier_output_g = self._classifier(fakeact_g)
                    fake_target = torch.argmax(tar_g, dim=1)
                    loss_class_g = classifier_loss(classifier_output_g, fake_target.detach())

                    if c1 is None:
                        y_fake = self._discriminator(torch.cat([fakeact_g, tar_g], dim=1))
                    else:
                        y_fake = self._discriminator(torch.cat([fakeact_g, c1_g, tar_g], dim=1))

                    if condvec is None:
                        loss_cond = 0
                    else:
                        loss_cond = self._cond_loss(fake_g, c1_g, m1_g)

                    loss_cluster = ((fakeact_g - centriod_g) ** 2).sum() / fakeact_g.shape[0]
                    loss_adv = -torch.mean(y_fake)
                    loss_g = loss_adv + self._alpha*loss_cond + self._beta*loss_cluster + self._gamma*loss_class_g

                    optimizerBackbone.zero_grad()
                    for target_value in self._target_value_list:
                        self._optimizerBranch_dict[target_value].zero_grad()

                    loss_g.backward()

                    optimizerBackbone.step()
                    for target_value in self._target_value_list:
                        self._optimizerBranch_dict[target_value].step()

            if self._verbose:
                print(f'Epoch {i + 1}, Loss G: {loss_g.detach().cpu(): .4f},'  # noqa: T001
                      f'Loss D: {loss_d.detach().cpu(): .4f}',
                      flush=True)

            loss_d_list.append(loss_d.item())
            loss_pen_list.append(pen.item())
            loss_class_r_list.append(loss_class_r.item())
            loss_adv_list.append(loss_adv.item())
            loss_cluster_list.append(loss_cluster.item())
            loss_class_g_list.append(loss_class_g.item())
            if condvec is None:
                pass
            else:
                loss_cond_list.append(loss_cond.item())

        if condvec is None:
            return {'loss_d': loss_d_list,
                    'loss_pen': loss_pen_list,
                    'loss_class_r': loss_class_r_list,
                    'loss_adv': loss_adv_list,
                    'loss_cluster': loss_cluster_list,
                    'loss_class_g': loss_class_g_list}
        else:
            return {'loss_d': loss_d_list,
                    'loss_pen': loss_pen_list,
                    'loss_class_r': loss_class_r_list,
                    'loss_adv': loss_adv_list,
                    'loss_cluster': loss_cluster_list,
                    'loss_class_g': loss_class_g_list,
                    'loss_cond': loss_cond_list}

    @random_state
    def sample(self, target_value, n):
        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)
            condvec = self._each_class_data_sampler_dict[target_value].sample_condvec(self._batch_size)

            if condvec is None:
                c1, m1, col, opt = None, None, None, None
                tar = self._target_one_hot[np.array(self._target) == target_value]
                tar = torch.from_numpy(tar).to(self._device)
                tar = tar[0].repeat(self._batch_size, 1)
                fakez = torch.cat([fakez, tar], dim=1)
            else:
                c1, m1, col, opt = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                tar = self._target_one_hot[np.array(self._target) == target_value]
                tar = torch.from_numpy(tar).to(self._device)
                tar = tar[0].repeat(self._batch_size, 1)
                fakez = torch.cat([fakez, c1, tar], dim=1)

            fake = self._backbone(fakez)
            fake = self._branch_dict[target_value](fake)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]
        return self._transformer.inverse_transform(data)

    def synthesis(self, minority_dict={}):
        if len(minority_dict) == 0:
            print('No input')
        else:
            output_minority = {}
            for target_value, num in minority_dict.items():
                output_minority[target_value] = self.sample(target_value, num)
        return output_minority

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._backbone is not None:
            self._backbone.to(self._device)
        if self._branch_dict is not None:
            for target_value in self._target_value_list:
                self._branch_dict[target_value].to(self._device)