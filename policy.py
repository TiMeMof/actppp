import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
import numpy as np
from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

from collections import OrderedDict
from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax
from robomimic.algo.diffusion_policy import replace_bn_with_gn, ConditionalUnet1D


from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel


class DiffusionPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()

        self.camera_names = args_override['camera_names']

        self.observation_horizon = args_override['observation_horizon'] ### TODO TODO TODO DO THIS
        self.action_horizon = args_override['action_horizon'] # apply chunk size
        self.prediction_horizon = args_override['prediction_horizon'] # chunk size
        self.num_inference_timesteps = args_override['num_inference_timesteps']
        self.ema_power = args_override['ema_power']
        self.lr = args_override['lr']
        self.weight_decay = 0

        self.num_kp = 32
        self.feature_dimension = 64
        self.ac_dim = args_override['action_dim']
        self.state_dim = args_override['state_dim'] # get from config instead of hardcoding
        self.obs_dim = self.feature_dimension * len(self.camera_names) + self.state_dim # camera features and proprio

        backbones = []
        pools = []
        linears = []
        for _ in self.camera_names:
            backbones.append(ResNet18Conv(**{'input_channel': 3, 'pretrained': False, 'input_coord_conv': False}))
            pools.append(SpatialSoftmax(**{'input_shape': [512, 15, 20], 'num_kp': self.num_kp, 'temperature': 1.0, 'learnable_temperature': False, 'noise_std': 0.0}))
            linears.append(torch.nn.Linear(int(np.prod([self.num_kp, 2])), self.feature_dimension))
        backbones = nn.ModuleList(backbones)
        pools = nn.ModuleList(pools)
        linears = nn.ModuleList(linears)
        
        backbones = replace_bn_with_gn(backbones) # TODO


        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=self.obs_dim*self.observation_horizon
        )

        nets = nn.ModuleDict({
            'policy': nn.ModuleDict({
                'backbones': backbones,
                'pools': pools,
                'linears': linears,
                'noise_pred_net': noise_pred_net
            })
        })

        nets = nets.float().cuda()
        ENABLE_EMA = True
        if ENABLE_EMA:
            ema = EMAModel(parameters=nets.parameters(), power=self.ema_power)
        else:
            ema = None
        self.nets = nets
        self.ema = ema

        # setup noise scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=50,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon'
        )

        n_parameters = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_parameters/1e6,))


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.nets.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


    def __call__(self, qpos, image, actions=None, is_pad=None):
        B = qpos.shape[0]
        if actions is not None: # training time
            nets = self.nets
            all_features = []
            for cam_id in range(len(self.camera_names)):
                cam_image = image[:, cam_id]
                cam_features = nets['policy']['backbones'][cam_id](cam_image)
                pool_features = nets['policy']['pools'][cam_id](cam_features)
                pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = nets['policy']['linears'][cam_id](pool_features)
                all_features.append(out_features)

            obs_cond = torch.cat(all_features + [qpos], dim=1)

            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=obs_cond.device)
            
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B,), device=obs_cond.device
            ).long()
            
            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps)
            
            # predict the noise residual
            noise_pred = nets['policy']['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)
            
            # L2 loss
            all_l2 = F.mse_loss(noise_pred, noise, reduction='none')
            loss = (all_l2 * ~is_pad.unsqueeze(-1)).mean()

            loss_dict = {}
            loss_dict['l2_loss'] = loss
            loss_dict['loss'] = loss

            if self.training and self.ema is not None:
                self.ema.step(nets)
            return loss_dict
        else: # inference time
            To = self.observation_horizon
            Ta = self.action_horizon
            Tp = self.prediction_horizon
            action_dim = self.ac_dim
            
            nets = self.nets
            if self.ema is not None:
                # 创建临时模型副本并应用EMA参数
                import copy
                nets = copy.deepcopy(self.nets)
                self.ema.copy_to(nets.parameters())
            
            all_features = []
            for cam_id in range(len(self.camera_names)):
                cam_image = image[:, cam_id]
                cam_features = nets['policy']['backbones'][cam_id](cam_image)
                pool_features = nets['policy']['pools'][cam_id](cam_features)
                pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = nets['policy']['linears'][cam_id](pool_features)
                all_features.append(out_features)

            obs_cond = torch.cat(all_features + [qpos], dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, Tp, action_dim), device=obs_cond.device)
            naction = noisy_action
            
            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = nets['policy']['noise_pred_net'](
                    sample=naction, 
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

            return naction

    def serialize(self):
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict):
        status = self.nets.load_state_dict(model_dict["nets"])
        print('Loaded model')
        if model_dict.get("ema", None) is not None:
            print('Loaded EMA')
            status_ema = self.ema.load_state_dict(model_dict["ema"])
            status = [status, status_ema]
        return status

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.vq = args_override['vq']
        self.use_ee = args_override['use_ee']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None, vq_sample=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            # num_queries就是chunk size大小
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            loss_dict = dict()
            a_hat, is_pad_hat, (mu, logvar), probs, binaries = self.model(qpos, image, env_state, actions, is_pad, vq_sample)
            if self.vq or self.model.encoder is None:
                total_kld = [torch.tensor(0.0)]
            else:
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            if self.vq:
                loss_dict['vq_discrepancy'] = F.l1_loss(probs, binaries, reduction='mean')

            if self.use_ee:
                # 左臂
                pos_l = a_hat[..., :3]          # (B, num_queries, 3)
                rot6d_l = a_hat[..., 3:9]       # (B, num_queries, 6)
                grip_l = a_hat[..., 9:10]       # (B, num_queries, 1)
                # 6D -> R
                rot6d_l = rot6d_l.reshape(-1, 6)         # (B*num_queries, 6)
                R_l = self.rot6d_to_matrix(rot6d_l)           # (B*num_queries, 3, 3)

                # 右臂
                pos_r = a_hat[..., 10:13]
                rot6d_r = a_hat[..., 13:19]
                grip_r = a_hat[..., 19:20]
                rot6d_r = rot6d_r.reshape(-1, 6)
                R_r = self.rot6d_to_matrix(rot6d_r)

                # 左臂 GT
                pos_l_gt   = actions[..., :3]
                quat_l_gt  = actions[..., 3:7]
                grip_l_gt  = actions[..., 7:8]
                # 右臂 GT
                pos_r_gt   = actions[..., 8:11]
                quat_r_gt  = actions[..., 11:15]
                grip_r_gt  = actions[..., 15:16]
                R_l_gt = self.quat_to_matrix(quat_l_gt.reshape(-1, 4))
                R_r_gt = self.quat_to_matrix(quat_r_gt.reshape(-1, 4))

                # pos loss
                loss_pos = F.l1_loss(pos_l, pos_l_gt) + F.l1_loss(pos_r, pos_r_gt)

                # gripper loss
                loss_grip = F.l1_loss(grip_l, grip_l_gt) + F.l1_loss(grip_r, grip_r_gt)

                # rot loss (Frobenius norm)
                loss_rot = self.rotation_matrix_loss(R_l, R_l_gt) + \
                        self.rotation_matrix_loss(R_r, R_r_gt)

                # 总损失
                loss_dict['pos'] = loss_pos
                loss_dict['rot'] = loss_rot*20
                loss_dict['grip']= loss_grip
                all_l1 = loss_dict['pos'] + loss_dict['rot'] + loss_dict['grip']
                l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
                loss_dict['l1'] = l1
                loss_dict['kl'] = total_kld[0]
                loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight

                return loss_dict
            else:
                all_l1 = F.l1_loss(actions, a_hat, reduction='none')
                l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
                loss_dict['l1'] = l1
                loss_dict['kl'] = total_kld[0]
                loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
                return loss_dict
        else: # inference time
            a_hat, _, (_, _), _, _ = self.model(qpos, image, env_state, vq_sample=vq_sample) # no action, sample from prior
            if self.use_ee:
                # 左臂
                pos_l = a_hat[..., :3]          # (B, num_queries, 3)
                rot6d_l = a_hat[..., 3:9]       # (B, num_queries, 6)
                grip_l = a_hat[..., 9:10]       # (B, num_queries, 1)
                # 6D -> R -> quat
                rot6d_l = rot6d_l.reshape(-1, 6)         # (B*num_queries, 6)
                R_l = self.rot6d_to_matrix(rot6d_l)           # (B*num_queries, 3, 3)
                quat_l = self.matrix_to_quaternion(R_l)        # (B*num_queries, 4)
                quat_l = quat_l.view(a_hat.shape[0], a_hat.shape[1], 4)  # reshape回 (B, num_queries, 4)
                quat_l = quat_l[..., [3, 0, 1, 2]]  # 变成 (w,x,y,z)

                # 右臂
                pos_r = a_hat[..., 10:13]
                rot6d_r = a_hat[..., 13:19]
                grip_r = a_hat[..., 19:20]
                rot6d_r = rot6d_r.reshape(-1, 6)
                R_r = self.rot6d_to_matrix(rot6d_r)
                quat_r = self.matrix_to_quaternion(R_r)
                quat_r = quat_r.view(a_hat.shape[0], a_hat.shape[1], 4)  # reshape回 (B, num_queries, 4)
                quat_r = quat_r[..., [3, 0, 1, 2]]  # (w,x,y,z)

                zeros = torch.zeros(a_hat.shape[0], a_hat.shape[1], 2, device=a_hat.device, dtype=a_hat.dtype)

                a_hat = torch.cat([pos_l, quat_l, grip_l, pos_r, quat_r, grip_r, zeros], dim=-1)

            return a_hat

    def configure_optimizers(self):
        return self.optimizer

    @torch.no_grad()
    def vq_encode(self, qpos, actions, is_pad):
        # 对动作进行向量量化编码
        actions = actions[:, :self.model.num_queries]
        is_pad = is_pad[:, :self.model.num_queries]

        _, _, binaries, _, _ = self.model.encode(qpos, actions, is_pad)

        return binaries
        
    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)
    
    @staticmethod
    def quat_to_matrix(quat):
        """
        quat: (B, 4) normalized quaternion
        return: (B, 3, 3) rotation matrix
        """
        quat = F.normalize(quat, dim=-1)
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        B = quat.size(0)
        R = torch.zeros((B, 3, 3), device=quat.device)
        R[:, 0, 0] = 1 - 2*(y**2 + z**2)
        R[:, 0, 1] = 2*(x*y - z*w)
        R[:, 0, 2] = 2*(x*z + y*w)
        R[:, 1, 0] = 2*(x*y + z*w)
        R[:, 1, 1] = 1 - 2*(x**2 + z**2)
        R[:, 1, 2] = 2*(y*z - x*w)
        R[:, 2, 0] = 2*(x*z - y*w)
        R[:, 2, 1] = 2*(y*z + x*w)
        R[:, 2, 2] = 1 - 2*(x**2 + y**2)
        return R

    
    @staticmethod
    def rotation_matrix_loss(R_pred, R_gt):
        # R_pred, R_gt: (B, 3, 3)
        loss = torch.norm(R_pred - R_gt, dim=(1,2))  # Frobenius norm
        return loss.mean()
    
    @staticmethod
    def rot6d_to_matrix(x):
        """
        Convert 6D rotation representation to 3x3 rotation matrix.
        Input: (B, 6) tensor
        Output: (B, 3, 3) rotation matrix
        """
        x = x.view(-1, 6)
        a1 = x[:, 0:3]
        a2 = x[:, 3:6]

        def normalize_vector(v, eps=1e-8):
            """ Normalize a batch of vectors """
            return v / (torch.norm(v, dim=-1, keepdim=True) + eps)
        b1 = normalize_vector(a1)
        b2 = normalize_vector(a2 - torch.sum(b1 * a2, dim=1, keepdim=True) * b1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)  # (B, 3, 3)

    @staticmethod
    def matrix_to_quaternion(R):
        """
        Convert rotation matrix to quaternion.
        Input: (B, 3, 3)
        Output: (B, 4) quaternions (x, y, z, w)
        """
        B = R.shape[0]
        quat = torch.zeros(B, 4, device=R.device)

        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        for i in range(B):
            if trace[i] > 0:
                s = torch.sqrt(trace[i] + 1.0) * 2
                quat[i, 3] = 0.25 * s
                quat[i, 0] = (R[i, 2, 1] - R[i, 1, 2]) / s
                quat[i, 1] = (R[i, 0, 2] - R[i, 2, 0]) / s
                quat[i, 2] = (R[i, 1, 0] - R[i, 0, 1]) / s
            else:
                # fallback cases, see: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
                if (R[i, 0, 0] > R[i, 1, 1]) and (R[i, 0, 0] > R[i, 2, 2]):
                    s = torch.sqrt(1.0 + R[i, 0, 0] - R[i, 1, 1] - R[i, 2, 2]) * 2
                    quat[i, 3] = (R[i, 2, 1] - R[i, 1, 2]) / s
                    quat[i, 0] = 0.25 * s
                    quat[i, 1] = (R[i, 0, 1] + R[i, 1, 0]) / s
                    quat[i, 2] = (R[i, 0, 2] + R[i, 2, 0]) / s
                elif R[i, 1, 1] > R[i, 2, 2]:
                    s = torch.sqrt(1.0 + R[i, 1, 1] - R[i, 0, 0] - R[i, 2, 2]) * 2
                    quat[i, 3] = (R[i, 0, 2] - R[i, 2, 0]) / s
                    quat[i, 0] = (R[i, 0, 1] + R[i, 1, 0]) / s
                    quat[i, 1] = 0.25 * s
                    quat[i, 2] = (R[i, 1, 2] + R[i, 2, 1]) / s
                else:
                    s = torch.sqrt(1.0 + R[i, 2, 2] - R[i, 0, 0] - R[i, 1, 1]) * 2
                    quat[i, 3] = (R[i, 1, 0] - R[i, 0, 1]) / s
                    quat[i, 0] = (R[i, 0, 2] + R[i, 2, 0]) / s
                    quat[i, 1] = (R[i, 1, 2] + R[i, 2, 1]) / s
                    quat[i, 2] = 0.25 * s

        # Normalize to unit quaternion
        quat = F.normalize(quat, dim=-1)
        return quat


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def normalize_quaternion(quat):
    """
    归一化四元数
    quat: (..., 4) tensor，四元数格式为 [w, x, y, z]
    """
    return F.normalize(quat, p=2, dim=-1)

def quaternion_dot_loss(pred_quat, target_quat):
    """
    计算四元数的点乘损失
    pred_quat: 预测的四元数 (..., 4)
    target_quat: 目标四元数 (..., 4)
    """
    # 归一化四元数
    pred_quat_norm = normalize_quaternion(pred_quat)
    target_quat_norm = normalize_quaternion(target_quat)
    
    # 计算点乘 (cosine similarity)
    dot_product = torch.sum(pred_quat_norm * target_quat_norm, dim=-1)
    
    # 由于四元数q和-q代表相同的旋转，取绝对值
    dot_product = torch.abs(dot_product)
    
    # 损失函数：1 - |dot_product| (越接近1越好)
    loss = 1.0 - dot_product
    return loss.mean()

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
