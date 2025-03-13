
import math
import clip 
import pytorch_lightning as pl

import torch
import torch.nn as nn 
import torch.nn.functional as F

from torch.distributions import MultivariateNormal, kl_divergence
from torch.optim import Adam

class MCIL(pl.LightningModule):
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = MCILModel(config["model"])

        self.lr = config["lr"]
        self.loss = nn.CrossEntropyLoss(reduction="none")

        self.binary_weight = config["binary_weight"]
        self.beta = config["beta"]
        self.std = config["std"]

        self.action_dim = config["model"]["action_dim"]
        self.log_scale_min = config["log_scale_min"]
        self.n_dist = config["model"]["n_dist"]
        self.num_classes = config["num_classes"]
        self.segmentation = config["model"]["segmentation"]
        self.only_stop = config["model"]["only_stop"]

        self.register_buffer("action_min_bound",(
            -torch.ones(self.action_dim)
            .to(self.device)
            .reshape(1, 1, len(config["act_min_bound"]))
            .unsqueeze(-1)
            .repeat(1, 1, 1, self.n_dist)
        ))

        self.register_buffer("action_max_bound",(
            torch.ones(self.action_dim)
            .to(self.device)
            .reshape(1, 1, len(config["act_min_bound"]))
            .unsqueeze(-1)
            .repeat(1, 1, 1, self.n_dist)
        ))

    def configure_optimizers(self):
        return Adam(self.model.parameters(),lr=self.lr)

    def training_step(self, data, data_id):
        actions, variational_param, prior_param = self.model(data)
        nll, kl = self.compute_elbo(actions, variational_param, prior_param, data)
        loss = nll + self.beta * kl

        info = {
            "loss": loss,
            "nll": nll,
            "kl": kl,
        }
        metrics = self.get_metrics(actions, data)
        info.update(metrics)

        for key, value in info.items():
            self.log(f"training/{key}",value)

        return loss

    def validation_step(self, data, data_id):
        actions, variational_param, prior_param = self.model(data)
        actions = self.model.decode(data, prior_param[0])
        metrics = self.get_metrics(actions, data)

        for key, value in metrics.items():
            self.log(f"validation/{key}",value, sync_dist=True)

    def get_metrics(self, actions, data):
        # Sample an action
        stop_prob = F.sigmoid(actions[-1])
        actions, binary_stops = self._sample_actions(actions)

        # Compute absolute difference
        B, T, _ = actions.shape
        mask = ~get_padding_mask(B, T, data["obs_length"], self.device)

        abs_diff = torch.abs(actions[:, :, :-1] - data["actions"][:, :, :-1])[
            mask
        ].mean()

        # Gripper accuracy
        pos_gripper_mask = torch.logical_and(
            data["actions"][:, :, 6] > 0, actions[:, :, 6] > 0
        )
        neg_gripper_mask = torch.logical_and(
            data["actions"][:, :, 6] < 0, actions[:, :, 6] < 0
        )
        gripper_accuracy = (pos_gripper_mask + neg_gripper_mask)[mask].float().mean()

        # Terminal accuracy
        if self.segmentation:
            d = data["obs_length"].device
            gt_termination = torch.arange(T).expand(B,T).to(d) == (data["obs_length"]-1).reshape(-1, 1)
            terminal_accuracy_per_timestep = (gt_termination==binary_stops)
            terminal_accuracy = terminal_accuracy_per_timestep.float().mean()

            indeces = torch.arange(B).to(data["obs_length"].device)
            stop_accuracy = terminal_accuracy_per_timestep[indeces,data["obs_length"]-1].float().mean()
            avg_stop_prob = stop_prob[mask].mean()
            avg_stop_prob_end = stop_prob[indeces,data["obs_length"]-1].mean()
        else:
            terminal_accuracy = 0
            avg_stop_prob = 0
            avg_stop_prob_end = 0

        # Compute based on language and goal
        if torch.any(data["mask"]):
            lang_padding_mask = mask[data["mask"]]
            lang_action_accuracy = torch.abs(
                actions[:, :, :-1] - data["actions"][:, :, :-1]
            )[data["mask"]][lang_padding_mask].mean()
            lang_gripper_accuracy = (
                (pos_gripper_mask + neg_gripper_mask)[data["mask"]][lang_padding_mask].float().mean()
            )
            if self.segmentation:
                lang_terminal_accuracy = terminal_accuracy_per_timestep[data["mask"]].float().mean()
            else:
                lang_terminal_accuracy = 0
        else:
            lang_action_accuracy = 0
            lang_gripper_accuracy = 0
            lang_terminal_accuracy = 0

        if not torch.all(data["mask"]):
            goal_padding_mask = mask[~data["mask"]]
            goal_action_accuracy = torch.abs(
                actions[:, :, :-1] - data["actions"][:, :, :-1]
            )[~data["mask"]][goal_padding_mask].mean()
            goal_gripper_accuracy = (
                (pos_gripper_mask + neg_gripper_mask)[~data["mask"]][goal_padding_mask].float().mean()
            )
            if self.segmentation:
                goal_terminal_accuracy = terminal_accuracy_per_timestep[~data["mask"]].float().mean()
            else:
                goal_terminal_accuracy = 0

        else:
            goal_action_accuracy = 0
            goal_gripper_accuracy = 0
            goal_terminal_accuracy = 0

        return {
            "action_accuracy": abs_diff,
            "gripper_accuracy": gripper_accuracy,
            "terminal_accuracy": terminal_accuracy,
            "lang_action_accuracy": lang_action_accuracy,
            "lang_gripper_accuracy": lang_gripper_accuracy,
            "lang_terminal_accuracy": lang_terminal_accuracy,
            "goal_action_accuracy": goal_action_accuracy,
            "goal_gripper_accuracy": goal_gripper_accuracy,
            "goal_terminal_accuracy": goal_terminal_accuracy,
            "avg_stop_prob": avg_stop_prob,
            "avg_stop_prob_end": avg_stop_prob_end
        }

    def compute_kl(self, variational, prior):
        variational_mvn = MultivariateNormal(variational[0], variational[1])
        prior_mvn = MultivariateNormal(prior[0], prior[1])
        kl = kl_divergence(variational_mvn, prior_mvn)
        return kl

    def compute_elbo(self, actions, variational, prior_param, data):
        B, T, _ = actions[0].shape
        mask = ~get_padding_mask(B, T, data["obs_length"], self.device)
        nll, _, _ = self.get_loss(actions, data, mask)
        kl = self.compute_kl(variational, prior_param).mean()

        return nll, kl

    def get_loss(self, actions, data, mask):
        # Parameters of the Mixture of logisitics distribution
        if self.segmentation and self.action_dim==7:
            log_mixt_probs, means, log_scales, binary_stop = actions
        else:
            log_mixt_probs, means, log_scales = actions

        B, T, _ = means.shape
        log_mixt_probs = log_mixt_probs.reshape(B, T, self.action_dim, self.n_dist)
        means = means.reshape(B, T, self.action_dim, self.n_dist)
        log_scales = log_scales.reshape(B, T, self.action_dim, self.n_dist)

        # Clamp scale
        log_scales = torch.clamp(log_scales, min=self.log_scale_min)

        # Copy actions for mixture
        actions = data["actions"].unsqueeze(-1).repeat(1, 1, 1, self.n_dist)

        centered_actions = actions - means
        inv_stdv = torch.exp(-log_scales)
        act_range = (self.action_max_bound - self.action_min_bound) / 2.0
        plus_in = inv_stdv * (centered_actions + act_range / (self.num_classes - 1))
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_actions - act_range / (self.num_classes - 1))
        cdf_min = torch.sigmoid(min_in)

        # Corner Cases
        log_cdf_plus = plus_in - F.softplus(
            plus_in
        )  # log probability for edge case of 0 (before scaling)
        log_one_minus_cdf_min = -F.softplus(
            min_in
        )  # log probability for edge case of 255 (before scaling)
        # Log probability in the center of the bin
        mid_in = inv_stdv * centered_actions
        log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)
        # Probability for all other cases
        cdf_delta = cdf_plus - cdf_min

        # Log probability
        log_probs = torch.where(
            actions < self.action_min_bound + 1e-3,
            log_cdf_plus,
            torch.where(
                actions > self.action_max_bound - 1e-3,
                log_one_minus_cdf_min,
                torch.where(
                    cdf_delta > 1e-5,
                    torch.log(torch.clamp(cdf_delta, min=1e-12)),
                    log_pdf_mid - np.log((self.num_classes - 1) / 2),
                ),
            ),
        )

        log_probs = log_probs + F.log_softmax(log_mixt_probs, dim=-1)
        action_loss = torch.sum(log_sum_exp(log_probs), dim=-1).float()

        if self.segmentation:
            prob_stop = F.sigmoid(binary_stop.squeeze(-1))
            stop_log_prob = torch.log(prob_stop)
            if self.only_stop:
                action_loss = torch.log(1-prob_stop)
                indeces = torch.arange(B).to(data["obs_length"].device)
                action_loss[indeces,data["obs_length"]-1] = torch.log(prob_stop[indeces,data["obs_length"]-1]) 
                action_log_prob = torch.zeros_like(stop_log_prob)
            else:
                action_loss += torch.log(1-prob_stop)
                indeces = torch.arange(B).to(data["obs_length"].device)
                action_loss[indeces,data["obs_length"]-1] += self.binary_weight*torch.log(prob_stop[indeces,data["obs_length"]-1])
                action_log_prob = log_probs.sum((2,3))
        else:
            stop_log_prob = 0
            action_log_prob = 0

        action_loss = -action_loss[mask].mean()

        return action_loss, action_log_prob, stop_log_prob

    def get_log_probs(self,data,device):
        self.to_device(data,device)
        actions, variational_param, prior_param = self.model(data)

        # Determine
        _, _, _, binary_stop = actions

        # Get log-likelihood
        B, T, _ = data["actions"].shape
        mask = ~get_padding_mask(B, T, data["obs_length"], self.device)
        _, action_log_probs, stop_log_prob = self.get_loss(actions,data,mask)

        return stop_log_prob, action_log_probs

    def _sample_actions(self, actions):
        if self.segmentation:
            logit_mixt_probs, means, log_scales, binary_stop = actions
        else:
            logit_mixt_probs, means, log_scales = actions

        B, T, _ = log_scales.shape
        log_scales = log_scales.reshape(B, T, self.action_dim, self.n_dist)
        means = means.reshape(B, T, self.action_dim, self.n_dist)
        logit_mixt_probs = logit_mixt_probs.reshape(B, T, self.action_dim, self.n_dist)

        r1, r2 = 1e-5, 1.0 - 1e-5
        temp = (r1 - r2) * torch.rand(means.shape, device=means.device) + r2
        temp = logit_mixt_probs - torch.log(-torch.log(temp))
        argmax = torch.argmax(temp, -1)

        dist = torch.eye(self.n_dist, device=self.device)[argmax]

        # Select scales and means
        log_scales = (dist * log_scales).sum(dim=-1)
        means = (dist * means).sum(dim=-1)

        # Inversion sampling for logistic mixture sampling
        scales = torch.exp(log_scales)  # Make positive
        u = (r1 - r2) * torch.rand(means.shape, device=means.device) + r2
        actions = means + scales * (torch.log(u) - torch.log(1.0 - u))

        # Sample binary-stops
        if self.segmentation:
            binary_stop = torch.round(F.sigmoid(binary_stop)).squeeze(-1)
        else:
            binary_stop = 0

        return actions, binary_stop

    def act(self, data, device, resample=False):
        self.to_device(data,device)
        actions, variational_param, prior_param = self.model(data)

        if resample:
            self.last_prior_param = prior_param

        prior_param = self.last_prior_param

        actions = self.model.decode(data, prior_param[0])

        actions, _ = self._sample_actions(actions)
        actions = actions[0][-1].reshape(1, -1)
        return actions

    def act_with_teacher(self, data):
        self.to_device(data)
        actions, variational_param, prior_param = self.model(data)

        if self.data_parallel:
            actions = self.model.module.decode(data, prior_param[0])
        else:
            actions = self.model.decode(data, prior_param[0])

        actions, _ = self._sample_actions(actions)
        actions[:, :, -1] = torch.where(actions[:, :, -1] > 0, 1, -1)
        return actions

    def to_device(self,data,device):
        for key, value in data.items():
            if isinstance(value,torch.Tensor):
                data[key] = value.to(device)


class MCILModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.action_dim = config["action_dim"]
        self.latent_dim = config["latent_dim"]
        self.lang_dim = config["lang_dim"]
        self.n_dist = config["n_dist"]
        self.segmentation = config["segmentation"]

        # Perceptual encoders
        self.img_encoder = ImageEncoderCalvin(False)
        self.gripper_encoder = GripperEncoderCalvin(False)

        # Trajectory Encoder
        self.encoder_input_dim = config["obs_dim"] + config["gripper_obs_dim"]

        self.trajectory_encoder = TrajectoryTransformer(
            self.encoder_input_dim,
            config["encoder"]["input_dim"],
            config["encoder"]["nhead"],
            config["encoder"]["hidden_dim"],
            config["encoder"]["num_layers"],
            config["encoder"]["dropout"],
            True,
            "complete",
            config["encoder"]["context_length"],
            config["encoder"]["use_positional_encodings"],
        )

        self.encoder_output = nn.Linear(
            config["encoder"]["input_dim"], config["latent_dim"] * 2
        )

        # Goal Encoder
        self.goal_emb_dim = config["goal_encoder"]["output_dimension"]
        self.lang_encoder = get_mlp(
            self.lang_dim + self.encoder_input_dim,
            config["goal_encoder"]["hidden_dimension"],
            config["goal_encoder"]["output_dimension"],
            config["goal_encoder"]["n_layers"],
        )

        self.state_encoder = get_mlp(
            self.encoder_input_dim * 2,
            config["goal_encoder"]["hidden_dimension"],
            config["goal_encoder"]["output_dimension"],
            config["goal_encoder"]["n_layers"],
        )

        # Prior
        self.lang_dim = 384
        self.prior = get_mlp(
            config["goal_encoder"]["output_dimension"],
            config["prior"]["hidden_dim"],
            self.latent_dim * 2,
            config["prior"]["num_layers"],
        )

        # Trajectory Decoder
        decoder_input_dim = (
            config["latent_dim"]
            + self.encoder_input_dim
            + config["goal_encoder"]["output_dimension"]
        )

        self.trajectory_decoder = TrajectoryTransformer(
            decoder_input_dim,
            config["decoder"]["input_dim"],
            config["decoder"]["nhead"],
            config["decoder"]["hidden_dim"],
            config["decoder"]["num_layers"],
            config["decoder"]["dropout"],
            False,
            "forward",
            config["decoder"]["context_length"],
            config["decoder"]["use_positional_encodings"],
        )

        self.decoder_means = nn.Linear(
            config["decoder"]["input_dim"], self.action_dim * self.n_dist
        )
        self.decoder_log_scales = nn.Linear(
            config["decoder"]["input_dim"], self.action_dim * self.n_dist
        )
        self.decoder_log_mixt_probs = nn.Linear(
            config["decoder"]["input_dim"], self.action_dim * self.n_dist
        )

        if self.segmentation:
            self.decoder_binary_stop = nn.Linear(config["decoder"]["input_dim"], 1)

    def forward(self, data):
        # Determine variational distribution
        variational_mean, variational_cov = self.encode(data)

        # Determine prior
        goal_emb = self.encode_goals(data)
        prior_mean, prior_cov = self.get_prior(goal_emb)

        # Prior
        latents = self.sample_latents(variational_mean, variational_cov)

        # Decoding
        actions = self.decode(data, latents)

        return actions, (variational_mean, variational_cov), (prior_mean, prior_cov)

    def decode(self, data, latents):
        trajectory = self._create_decoder_seq(data, latents)
        trajectory = self.trajectory_decoder(trajectory, data["obs_length"])
        means = self.decoder_means(trajectory)
        log_scales = self.decoder_log_scales(trajectory)
        log_mixt_probs = self.decoder_log_mixt_probs(trajectory)
        actions = (log_mixt_probs, means, log_scales)

        if self.segmentation:
            binary_stop = self.decoder_binary_stop(trajectory)
            actions = (log_mixt_probs, means, log_scales, binary_stop)
            return actions

        return actions

    def prior_inference(self, data):
        self._build_obs_encoder(data)
        prior_mean, prior_cov = self.compute_prior(data)
        samples = self.sample_latents(prior_mean, prior_cov)
        actions = self.decode(data, samples)
        return actions

    def encode(self, data):
        trajectory = self._build_obs_encoder(data)
        encoding = self.trajectory_encoder(trajectory, data["obs_length"])
        encoding = self.encoder_output(encoding)
        mean, cov = encoding[:, : self.latent_dim], F.softplus(
            encoding[:, self.latent_dim :]
        )
        cov = get_cov(cov)
        return mean, cov

    def _create_decoder_seq(self, data, latents):
        _, T, _ = data["encoder_obs"].shape
        obs = data["encoder_obs"]
        latents = latents = latents.unsqueeze(1).repeat(1, T, 1)
        goals = data["goals"].unsqueeze(1).repeat(1, T, 1)
        trajectory = torch.cat([obs, latents, goals], dim=-1)

        return trajectory

    def _build_obs_encoder(self, data):

        # Encode RGB image
        B, T, H, W, C = data["img_obs"].shape
        img_encodings = self.img_encoder(
            data["img_obs"].reshape(B * T, W, H, C).permute(0, 3, 1, 2)
        ).reshape(B, T, -1)
        data["img_encodings"] = img_encodings

        # Encode Gripper Image
        B, T, H, W, C = data["gripper_obs"].shape
        gripper_encodings = self.gripper_encoder(
            data["gripper_obs"].reshape(B * T, W, H, C).permute(0, 3, 1, 2)
        ).reshape(B, T, -1)
        data["gripper_encodings"] = gripper_encodings

        data["encoder_obs"] = torch.cat([img_encodings, gripper_encodings], dim=-1)

        return data["encoder_obs"]

    def encode_goals(self, data):

        B, T, _ = data["encoder_obs"].shape
        mask = data["mask"]
        goal_emb = torch.zeros(B, self.goal_emb_dim).to(data["encoder_obs"].device)

        # Embed goal images
        if not torch.all(mask):
            obs_length = data["obs_length"][~mask]
            start_states_goal = data["encoder_obs"][:, 0][~mask]
            goal_states = data["encoder_obs"][~mask]
            B_S, _, _ = goal_states.shape
            goal_states = goal_states[torch.arange(0, B_S), obs_length - 1]
            state_goals = self.state_encoder(
                torch.cat([start_states_goal, goal_states], dim=-1)
            )
            goal_emb[~mask] = state_goals

        # Embed instructions
        if torch.any(mask):
            start_states_lang = data["encoder_obs"][:, 0][mask]
            instructions = data["instructions"].float()[mask]
            lang_goals = self.lang_encoder(
                torch.cat([start_states_lang, instructions.squeeze(1)], dim=-1)
            )
            goal_emb[mask] = lang_goals

        data["goals"] = goal_emb

        return goal_emb

    def sample_latents(self, mean, cov):
        mvn = MultivariateNormal(mean, cov)
        samples = mvn.rsample()
        return samples

    def compute_prior(self, data):
        if self.dataset == "alfred":
            data["instructions"] = data["instructions"].unsqueeze(1).float()
        elif self.dataset == "calvin":
            data["instructions"] = data["instructions"].float()

        encoding = self.prior(
            torch.cat(
                [data["instructions"].squeeze(1), data["encoder_obs"][:, 0]], dim=-1
            )
        )
        mean, cov = encoding[:, : self.latent_dim], F.softplus(
            encoding[:, self.latent_dim :]
        )
        cov = get_cov(cov)
        return mean, cov

    def get_prior(self, goal_emb):
        latent_embeddings = self.prior(goal_emb)
        mean, cov = latent_embeddings[:, : self.latent_dim], F.softplus(
            latent_embeddings[:, self.latent_dim :]
        )
        cov = get_cov(cov)
        return mean, cov




def get_mlp(
    input_dimension: int, hidden_dimension: int, output_dimension: int, n_layers: int
) -> nn.Module:
    assert n_layers > 0, "Number of layers must be a positive integer"
    layers = [nn.Linear(input_dimension, hidden_dimension), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dimension, hidden_dimension), nn.ReLU()]
    layers.append(nn.Linear(hidden_dimension, output_dimension))
    return nn.Sequential(*layers)


def get_device(config: dict) -> torch.device:
    device = config["device"] if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def get_positional_embeddings(num_embeddings: int, n: int, d: int):
    embeddings = nn.Embedding(num_embeddings, d)
    matrix = (
        torch.arange(0, num_embeddings).reshape(-1, 1)
        * 1
        / (n ** (2 * torch.arange(0, d / 2) / d))
    )
    even = torch.arange(0, d, 2)
    odd = torch.arange(1, d, 2)
    embeddings.weight.requires_grad = False
    embeddings.weight[:, even] = torch.sin(matrix)
    embeddings.weight[:, odd] = torch.cos(matrix)
    return embeddings


def get_n_parameters(model: nn.Module):
    return sum([p.numel() for p in model.parameters()])


def get_tensor_device(tensor: torch.tensor):
    return torch.device(f"cuda:{tensor.get_device()}")


def get_padding_mask(batch_size: int, length: int, seq_length: torch.tensor, device):
    padding_mask = torch.arange(0, length).expand(batch_size, length).to(
        device
    ) > seq_length.reshape(-1, 1)
    return padding_mask


def get_sequence_mask(
    T: int, context_length: int, obs_length: torch.tensor, n_head: int, device
) -> torch.Tensor:
    """
    Get a mask that prevents the model to look forward in time
    """
    mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(device)

    if context_length >= 0:
        mask = mask.unsqueeze(0).repeat(len(obs_length), 1, 1)
        for k, size in enumerate(obs_length):
            for i in range(size):
                if i > context_length:
                    mask[k, i, : i - context_length] = 1
        mask = torch.repeat_interleave(mask, n_head, dim=0)
    return mask


def get_cov(cov: torch.Tensor):
    # Transform a batch of vectors to diagonal covariance matrices
    B, D = cov.shape
    extended_cov = torch.zeros(B, D, D).to(cov.device)
    extended_cov[:, torch.arange(D), torch.arange(D)] = cov
    return extended_cov


def log_sum_exp(x):
    """numerically stable log_sum_exp implementation that prevents overflow"""
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def get_vlm(name, pretrained, download_root, device, freeze=True):
    model, transform = clip.load(name=name, device=device, download_root=download_root)

    if not pretrained:
        model.initialize_parameters()

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model, transform


class FocalLoss(nn.Module):

    def __init__(self, gamma: float):

        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, gt: torch.Tensor):
        """
        logits: BxC
        gt: B
        """
        prob = F.softmax(logits, dim=-1)
        prob = torch.gather(prob, dim=1, index=gt)
        return -((1 - prob) ** self.gamma) * prob.log()


class GripperEncoderCalvin(nn.Module):
    def __init__(self, depth):
        super().__init__()

        self.conv1 = nn.Conv2d(3 + depth, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute the output size of the last convolutional layer
        self.last_conv_output_size = 64 * 7 * 7

        self.fc1 = nn.Linear(self.last_conv_output_size, 256)
        self.fc2 = nn.Linear(256, 32)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Spatial softmax
        x = F.softmax(x.view(x.size(0), x.size(1), -1), dim=-1)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class ImageEncoderCalvin(nn.Module):
    def __init__(self, depth):
        super().__init__()

        self.conv1 = nn.Conv2d(3 + depth, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute the output size of the last convolutional layer
        self.last_conv_output_size = 64 * 21 * 21

        self.fc1 = nn.Linear(self.last_conv_output_size, 512)
        self.fc2 = nn.Linear(512, 64)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Spatial softmax
        x = F.softmax(x.view(x.size(0), x.size(1), -1), dim=-1)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class DownConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        group_size=32,
        activation="relu",
    ):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.GroupNorm(group_size, out_channels)
        self.activation = get_activation(activation)
        self.pooling = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Conv layers
        x = self.conv2d(x)
        # Normalization
        x = self.norm(x)
        # Activation
        x = self.activation(x)
        # Downsampling
        x = self.pooling(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[: x.size(1)].unsqueeze(0)
        return self.dropout(x)


class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """

    def __init__(
        self,
        num_channels,
        eps=1e-5,
        affine=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs)
            )
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x**2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out = out * self.weight
            out = out + self.bias

        return out


class TrajectoryConvolution(nn.Module):
    def __init__(self, input_dim, d_dim, context_length, n_layers):
        super().__init__()

        self.projection = nn.Linear(input_dim, d_dim)

        self.n_layers = n_layers
        self.convolutional_encoder = nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        self.relu = nn.ReLU()

        for layer in range(n_layers):
            layer = torch.nn.Conv1d(
                in_channels=d_dim,
                out_channels=d_dim,
                kernel_size=context_length,
                stride=1,
                padding=context_length - 1,
                padding_mode="zeros",
            )

            self.convolutional_encoder.append(layer)
            self.layer_norm.append(LayerNorm(d_dim))

    def forward(self, seq, obs_length, inference=False):
        trajectory = self._create_traj(seq)  # B,D,T

        for i in range(self.n_layers):
            _, _, T = trajectory.shape

            first_trajectory = self.convolutional_encoder[i](
                trajectory
            )  # B,D,T+context_length
            second_trajectory = first_trajectory[:, :, :T]
            third_trajectory = F.relu(self.layer_norm[i](second_trajectory))
            trajectory = third_trajectory.clone()

        return trajectory.permute(0, 2, 1)

    def _create_traj(self, seq):
        inputs = list(seq.values())
        return self.projection(torch.cat(inputs, dim=-1)).permute(0, 2, 1)


class TrajectoryTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_dim,
        n_head,
        hidden_dim,
        num_layers,
        dropout,
        aggregate,
        mask_type,
        context_length,
        use_positional_encoding,
    ):
        super().__init__()

        self.aggregate = aggregate
        self.mask_type = mask_type
        self.use_positional_encoding = use_positional_encoding
        self.context_length = context_length
        self.input_projection = nn.Linear(input_dim, d_dim)
        self.projection = nn.Linear(input_dim, d_dim)
        self.positional_encoding = PositionalEncoding(d_dim, dropout=dropout)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_dim,
            nhead=n_head,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer, num_layers=num_layers
        )
        self.n_head = n_head

    def forward(self, trajectory, obs_length):

        B, T, _ = trajectory.shape
        trajectory_mask = get_padding_mask(B, T, obs_length, trajectory.device)
        trajectory = self.input_projection(trajectory)

        if self.mask_type == "forward":
            key_mask = get_sequence_mask(
                T, self.context_length, obs_length, self.n_head, trajectory.device
            )

        elif self.mask_type == "complete":
            key_mask = torch.zeros(T, T).bool().to(trajectory.device)
        else:
            raise NotImplementedError("This mask type is not implemented")

        if self.use_positional_encoding:
            trajectory = self.positional_encoding(trajectory)

        trajectory = self.transformer_encoder(
            trajectory, mask=key_mask, src_key_padding_mask=trajectory_mask
        )

        if self.aggregate:
            trajectory[trajectory_mask] = 0
            trajectory = trajectory.sum(dim=1) / (
                T - trajectory_mask.sum(dim=1)
            ).reshape(-1, 1)

        return trajectory


def get_activation(fn_name: str):
    if fn_name == "relu":
        return nn.ReLU()
    else:
        raise NotImplementedError(f"Activation function {fn_name} not implemented")


# Code taken from https://github.com/mila-iqia/babyai

def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

