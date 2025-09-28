# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np

import IPython
e = IPython.embed


# VAE 中 的重参数化技巧
def reparametrize(mu, logvar):
    # 对数方差除以2，然后对结果取指数，得到方差的平方根，即标准差
    std = logvar.div(2).exp()
    # 生成随机噪声
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


# 生成正弦波编码表
# n_position：表示编码表中的位置数，即序列的最大长度。
# d_hid：表示隐藏层的维度，即编码向量的大小。
# 返回一个形状为 (1, n_position, d_hid) 的张量，包含了位置编码信息。
def get_sinusoid_encoding_table(n_position, d_hid):
    # 得到正弦波编码表
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, vq, vq_class, vq_dim, action_dim, use_ee):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        self.vq, self.vq_class, self.vq_dim = vq, vq_class, vq_dim
        self.state_dim, self.action_dim, self.use_ee = state_dim, action_dim, use_ee
        # 隐藏层维度，通常与 transformer 的 d_model 相同
        # d_model 是 transformer 中的一个重要参数，表示输入和输出的特征维度。
        hidden_dim = transformer.d_model
        # 把 transformer 的输出转成机器人动作向量。
        if self.use_ee:
            self.action_head = nn.Linear(hidden_dim, (3+6+1)*2)
        else:
            self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        # 这里的embedding层用于存储查询向量，这些查询向量在Transformer的解码器部分用于生成最终的输出。
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 64 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding

        print(f'Use VQ: {self.vq}, {self.vq_class}, {self.vq_dim}')
        if self.vq:
            self.latent_proj = nn.Linear(hidden_dim, self.vq_class * self.vq_dim)
        else:
            self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        if self.vq:
            self.latent_out_proj = nn.Linear(self.vq_class * self.vq_dim, hidden_dim)
        else:
            self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent

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

    # 实现了 VAE 编码器的功能，将输入数据（如机器人的位置 qpos 和动作序列 actions）编码为潜在空间的表示。
    def encode(self, qpos, actions=None, is_pad=None, vq_sample=None):
        bs, _ = qpos.shape
        if self.encoder is None:
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)
            probs = binaries = mu = logvar = None
        else:
            # cvae encoder
            is_training = actions is not None # train or val
            ### Obtain latent z from action sequence
            if is_training:
                # project action sequence to embedding dim, and concat with a CLS token
                action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
                qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
                qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
                cls_embed = self.cls_embed.weight # (1, hidden_dim)
                cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
                encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
                encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
                # do not mask cls token
                cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
                is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
                # obtain position embedding
                pos_embed = self.pos_table.clone().detach()
                pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
                # query model
                encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
                encoder_output = encoder_output[0] # take cls output only
                latent_info = self.latent_proj(encoder_output)
                
                if self.vq:
                    logits = latent_info.reshape([*latent_info.shape[:-1], self.vq_class, self.vq_dim])
                    probs = torch.softmax(logits, dim=-1)
                    binaries = F.one_hot(torch.multinomial(probs.view(-1, self.vq_dim), 1).squeeze(-1), self.vq_dim).view(-1, self.vq_class, self.vq_dim).float()
                    binaries_flat = binaries.view(-1, self.vq_class * self.vq_dim)
                    probs_flat = probs.view(-1, self.vq_class * self.vq_dim)
                    straigt_through = binaries_flat - probs_flat.detach() + probs_flat
                    latent_input = self.latent_out_proj(straigt_through)
                    mu = logvar = None
                else:
                    probs = binaries = None
                    mu = latent_info[:, :self.latent_dim]
                    logvar = latent_info[:, self.latent_dim:]
                    latent_sample = reparametrize(mu, logvar)
                    latent_input = self.latent_out_proj(latent_sample)

            else:
                mu = logvar = binaries = probs = None
                if self.vq:
                    latent_input = self.latent_out_proj(vq_sample.view(-1, self.vq_class * self.vq_dim))
                else:
                    latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
                    latent_input = self.latent_out_proj(latent_sample)

        return latent_input, probs, binaries, mu, logvar

    def forward(self, qpos, image, env_state, actions=None, is_pad=None, vq_sample=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        latent_input, probs, binaries, mu, logvar = self.encode(qpos, actions, is_pad, vq_sample)

        # cvae decoder
        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[cam_id](image[:, cam_id])
                features = features[0] # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        if self.use_ee:
            a_tmp = self.action_head(hs)
            
            # 左臂
            pos_l = a_tmp[..., :3]          # (B, num_queries, 3)
            rot6d_l = a_tmp[..., 3:9]       # (B, num_queries, 6)
            grip_l = a_tmp[..., 9:10]       # (B, num_queries, 1)
            # 6D -> R -> quat
            rot6d_l = rot6d_l.reshape(-1, 6)         # (B*num_queries, 6)
            R = self.rot6d_to_matrix(rot6d_l)           # (B*num_queries, 3, 3)
            quat_l = self.matrix_to_quaternion(R)        # (B*num_queries, 4)
            quat_l = quat_l.view(hs.shape[0], hs.shape[1], 4)  # reshape回 (B, num_queries, 4)
            quat_l = quat_l[..., [3, 0, 1, 2]]  # 变成 (w,x,y,z)

            # 右臂
            pos_r = a_tmp[..., 10:13]
            rot6d_r = a_tmp[..., 13:19]
            grip_r = a_tmp[..., 19:20]
            rot6d_r = rot6d_r.reshape(-1, 6)
            R = self.rot6d_to_matrix(rot6d_r)
            quat_r = self.matrix_to_quaternion(R)
            quat_r = quat_r.view(hs.shape[0], hs.shape[1], 4)  # reshape回 (B, num_queries, 4)
            quat_r = quat_r[..., [3, 0, 1, 2]]  # (w,x,y,z)

            # zero数组
            zeros = torch.zeros(hs.shape[0], hs.shape[1], 2, device=hs.device, dtype=hs.dtype)

            # 拼接成最终输出
            a_hat = torch.cat([pos_l, quat_l, grip_l, pos_r, quat_r, grip_r, zeros], dim=-1)
        else:
            a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar], probs, binaries



class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim) # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + state_dim
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=self.action_dim, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

# TransformerEncoder
# ├── TransformerEncoderLayer (x4)
# │   └── 具体结构见transformer.py
# └── 最终LayerNorm (如果normalize_before=True)

    # 创建编码器层
    '''
    d_model,               # 输入维度
    nhead,                # 注意力头数
    dim_feedforward,      # 前馈网络维度
    dropout,              # dropout率
    activation,           # 激活函数（relu）
    normalize_before      # 位置归一化选项
    '''
    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    # 构建完整编码器
    '''
    encoder_layer,         # 编码器层
    num_encoder_layers,    # 层数
    encoder_norm           # 归一化层
    '''
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build_vae(args):

    state_dim = args.state_dim
    use_ee = args.use_ee
    print("-----------------------------state_dim:::::", state_dim)
    print("-----------------------------use_ee:::::", use_ee)
    # state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    transformer = build_transformer(args)

    if args.no_encoder:
        encoder = None
    else:
        # encoder = build_transformer(args)
        encoder = build_encoder(args)

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=args.state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        vq=args.vq,
        vq_class=args.vq_class,
        vq_dim=args.vq_dim,
        action_dim=args.action_dim,
        use_ee=args.use_ee
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_cnnmlp(args):
    state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

