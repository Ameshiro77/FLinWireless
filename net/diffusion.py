import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import (
    cosine_beta_schedule,
    linear_beta_schedule,
    vp_beta_schedule,
    extract,
    Losses
)
from utils import Progress, Silent


class Diffusion(nn.Module):
    def __init__(self, state_dim, action_dim, model, max_action,
                 beta_schedule='linear', n_timesteps=100,
                 loss_type='l1', clip_denoised=False, predict_epsilon=True):
        super(Diffusion, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.model = model

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # log calculation clipped because the posterior variance
        # is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    def predict_start_from_noise(self, x_t, t, noise):
        """
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))

        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t, s):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(self, state, shape, verbose=False, return_diffusion=False):
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        if return_diffusion:
            diffusion = [x]
        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)
            progress.update({'t': i})
            if return_diffusion:
                diffusion.append(x)
        progress.close()
        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x, None

    def sample(self, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action, t_step_res = self.p_sample_loop(state, shape, *args, **kwargs)
        if t_step_res:
            return F.softmax(action, dim=-1), F.softmax(t_step_res, dim=-1)
        else:
            # return action
            return F.softmax(action, dim=-1)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.model(x_noisy, t, state)
       
        assert noise.shape == x_recon.shape
        if self.predict_epsilon:
            return self.loss_fn(x_recon, noise, weights)
        else:
            return self.loss_fn(x_recon, x_start, weights)

    def loss(self, x, state, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, t, weights)

    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)

if __name__ == '__main__':
    # create actor
    from model import MLP
    state_shape = 60
    action_shape = 10
    actor_net = MLP(state_dim=state_shape,
                    action_dim=action_shape,
                    hidden_dim=256)
    actor = Diffusion(state_dim=state_shape, action_dim=action_shape, model=actor_net,
                      max_action=10, beta_schedule='linear', n_timesteps=100).cuda()
    actor_optim = torch.optim.Adam(actor.parameters(),
                                   lr=0.001,
                                   weight_decay=0.01)

    states = torch.zeros((10, state_shape)).cuda()*1000   # 增大方差
    actions = torch.randint(0, 2, (10, action_shape), dtype=torch.float32).cuda()


    # print("original:",actions)
    # print("before trian:",actor.sample(states))

    # 训练循环
    for epoch in range(10):  # 训练 1000 个 epoch
        actor_optim.zero_grad()  # 清零梯度
        # loss = actor.loss(actions, states)  # 计算损失
        action = actor.sample(states)
        #loss.backward()  # 反向传播
        # print(loss)
        dist = torch.distributions.Categorical(action)
        e = dist.entropy().mean()
        print(e)
        e.backward()
        for name, param in actor.named_parameters():
            if param.grad is None:
                print(f"⚠️ WARNING: No gradient for {name}")
            else:
                print(f"✅ Gradient exists for {name}, mean grad: {param.grad.mean()}")
        exit()

        actor_optim.step()  # 更新参数
