""" PyTorch Lightning Trainer for MCIL policy"""

import math

import clip
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal, kl_divergence
from torch.optim import Adam
from typing import Tuple
import clip


class MCIL(pl.LightningModule):
    """
    MCIL (Multi Context Imitation Learning) LightningModule.

    Args:
        config (dict): Configuration dictionary.

    Attributes:
        model (MCILModel): The MCIL model.
        lr (float): The learning rate.
        loss (nn.CrossEntropyLoss): The loss function.
        binary_weight (float): The weight for binary stop.
        beta (float): The weight for the KL divergence term.
        std (float): The standard deviation.
        action_dim (int): The dimension of the action space.
        log_scale_min (float): The minimum log scale value.
        n_dist (int): The number of distributions.
        num_classes (int): The number of action classes.
        segmentation (bool): Flag indicating whether to use segmentation.
        only_stop (bool): Flag indicating whether to use only stop action.
        action_min_bound (torch.Tensor): The minimum bound for actions.
        action_max_bound (torch.Tensor): The maximum bound for actions.
    """

    def __init__(self, config: dict) -> None:
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

        self.register_buffer(
            "action_min_bound",
            (
                -torch.ones(self.action_dim)
                .to(self.device)
                .reshape(1, 1, len(config["act_min_bound"]))
                .unsqueeze(-1)
                .repeat(1, 1, 1, self.n_dist)
            ),
        )

        self.register_buffer(
            "action_max_bound",
            (
                torch.ones(self.action_dim)
                .to(self.device)
                .reshape(1, 1, len(config["act_min_bound"]))
                .unsqueeze(-1)
                .repeat(1, 1, 1, self.n_dist)
            ),
        )

    def configure_optimizers(self) -> torch.optim.Adam:
        """
        Configures the optimizer for the model.

        Returns:
            torch.optim.Adam: The Adam optimizer with the specified learning rate.
        """
        return Adam(self.model.parameters(), lr=self.lr)

    def training_step(self, data: dict, data_id: int) -> torch.Tensor:
        """
        Perform a single training step.

        Args:
            data (dict): The input data for the training step.
            data_id (int): The ID of the data.

        Returns:
            torch.Tensor: The loss value for the training step.
        """
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
            self.log(f"training/{key}", value)

        return loss

    def validation_step(self, data: dict, data_id: int) -> None:
        """
        Perform a validation step.

        Args:
            data (dict): The input data for the validation step.
            data_id (int): The ID of the data.

        Returns:
            None
        """
        actions, variational_param, prior_param = self.model(data)
        actions = self.model.decode(data, prior_param[0])
        metrics = self.get_metrics(actions, data)

        for key, value in metrics.items():
            self.log(f"validation/{key}", value, sync_dist=True)

    def get_metrics(self, actions:torch.Tensor, data:dict)->dict:
        """
        Compute various metrics based on the actions and input data.

        Args:
            actions (torch.Tensor): The predicted actions.
            data (dict): The input data.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
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
            gt_termination = torch.arange(T).expand(B, T).to(d) == (
                data["obs_length"] - 1
            ).reshape(-1, 1)
            terminal_accuracy_per_timestep = gt_termination == binary_stops
            terminal_accuracy = terminal_accuracy_per_timestep.float().mean()

            indeces = torch.arange(B).to(data["obs_length"].device)
            stop_accuracy = (
                terminal_accuracy_per_timestep[indeces, data["obs_length"] - 1]
                .float()
                .mean()
            )
            avg_stop_prob = stop_prob[mask].mean()
            avg_stop_prob_end = stop_prob[indeces, data["obs_length"] - 1].mean()
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
                (pos_gripper_mask + neg_gripper_mask)[data["mask"]][lang_padding_mask]
                .float()
                .mean()
            )
            if self.segmentation:
                lang_terminal_accuracy = (
                    terminal_accuracy_per_timestep[data["mask"]].float().mean()
                )
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
                (pos_gripper_mask + neg_gripper_mask)[~data["mask"]][goal_padding_mask]
                .float()
                .mean()
            )
            if self.segmentation:
                goal_terminal_accuracy = (
                    terminal_accuracy_per_timestep[~data["mask"]].float().mean()
                )
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
            "avg_stop_prob_end": avg_stop_prob_end,
        }

    def compute_kl(self, variational: Tuple[torch.Tensor, torch.Tensor], prior: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Compute the KL divergence between the variational and prior distributions.

        Args:
            variational (tuple): Tuple containing the mean and covariance of the variational distribution.
            prior (tuple): Tuple containing the mean and covariance of the prior distribution.

        Returns:
            torch.Tensor: The KL divergence.
        """
        variational_mvn = MultivariateNormal(variational[0], variational[1])
        prior_mvn = MultivariateNormal(prior[0], prior[1])
        kl = kl_divergence(variational_mvn, prior_mvn)
        return kl

    def compute_elbo(self, actions: torch.Tensor, variational: Tuple[torch.Tensor, torch.Tensor], prior_param: Tuple[torch.Tensor, torch.Tensor], data: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the evidence lower bound (ELBO) for the given actions and input data.

        Args:
            actions (torch.Tensor): The predicted actions.
            variational (tuple): Tuple containing the mean and covariance of the variational distribution.
            prior_param (tuple): Tuple containing the mean and covariance of the prior distribution.
            data (dict): The input data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The NLL (negative log-likelihood) and KL (KL divergence) values.
        """
        B, T, _ = actions[0].shape
        mask = ~get_padding_mask(B, T, data["obs_length"], self.device)
        nll, _, _ = self.get_loss(actions, data, mask)
        kl = self.compute_kl(variational, prior_param).mean()

        return nll, kl

    def compute_elbo(self, actions: torch.Tensor, variational: Tuple[torch.Tensor, torch.Tensor], prior_param: Tuple[torch.Tensor, torch.Tensor], data: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the evidence lower bound (ELBO) for the given actions and input data.

        Args:
            actions (torch.Tensor): The predicted actions.
            variational (tuple): Tuple containing the mean and covariance of the variational distribution.
            prior_param (tuple): Tuple containing the mean and covariance of the prior distribution.
            data (dict): The input data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The NLL (negative log-likelihood) and KL (KL divergence) values.
        """
        B, T, _ = actions[0].shape
        mask = ~get_padding_mask(B, T, data["obs_length"], self.device)
        nll, _, _ = self.get_loss(actions, data, mask)
        kl = self.compute_kl(variational, prior_param).mean()
        return nll, kl

    def get_log_probs(self, data: dict, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the log probabilities of actions and stop signal for the given data.

        Args:
            data (dict): The input data.
            device (str): The device to perform the calculations on.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the stop signal log probability and the action log probabilities.
        """
        self.to_device(data, device)
        actions, variational_param, prior_param = self.model(data)

        # Determine
        _, _, _, binary_stop = actions

        # Get log-likelihood
        B, T, _ = data["actions"].shape
        mask = ~get_padding_mask(B, T, data["obs_length"], self.device)
        _, action_log_probs, stop_log_prob = self.get_loss(actions, data, mask)

        return stop_log_prob, action_log_probs

    def _sample_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Samples actions from the given input tensor.

        Args:
            actions (torch.Tensor): Input tensor containing logit mixt probabilities, means, log scales.

        Returns:
            torch.Tensor: Sampled actions and binary stop.

        """
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
        binary_stop = 0

        return actions, binary_stop

    def act(self, data: dict, device: str, resample: bool) -> torch.Tensor:
        """
        Perform an action based on the given data.

        Args:
            data (dict): The input data.
            device (str): The device to perform the action on.
            resample (bool): Whether to resample the prior parameters.

        Returns:
            torch.Tensor: The resulting action.
        """
        self.to_device(data, device)
        actions, variational_param, prior_param = self.model(data)

        if resample:
            self.last_prior_param = prior_param

        prior_param = self.last_prior_param

        actions = self.model.decode(data, prior_param[0])

        actions, _ = self._sample_actions(actions)
        actions = actions[0][-1].reshape(1, -1)
        return actions

    def to_device(self, data: dict, device: str) -> None:
        """
        Move the tensors in the given data dictionary to the specified device.

        Args:
            data (dict): The dictionary containing the tensors to be moved.
            device (str): The device to move the tensors to.

        Returns:
            None
        """
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(device)


class MCILModel(nn.Module):
    """
    MCILModel is a PyTorch module that represents the MCIL (Multi Context Imitation Learning) model.

    Args:
        config (dict): A dictionary containing the configuration parameters for the model.

    Attributes:
        action_dim (int): The dimension of the action space.
        latent_dim (int): The dimension of the latent space.
        lang_dim (int): The dimension of the language space.
        n_dist (int): The number of distributions.
        segmentation (bool): A flag indicating whether segmentation is enabled.
        img_encoder (ImageEncoderCalvin): The perceptual encoder for RGB images.
        gripper_encoder (GripperEncoderCalvin): The perceptual encoder for gripper images.
        encoder_input_dim (int): The input dimension of the trajectory encoder.
        trajectory_encoder (TrajectoryTransformer): The trajectory encoder.
        encoder_output (nn.Linear): The linear layer for the encoder output.
        goal_emb_dim (int): The dimension of the goal embedding.
        lang_encoder (nn.Module): The language encoder.
        state_encoder (nn.Module): The state encoder.
        prior (nn.Module): The prior network.
        trajectory_decoder (TrajectoryTransformer): The trajectory decoder.
        decoder_means (nn.Linear): The linear layer for the decoder means.
        decoder_log_scales (nn.Linear): The linear layer for the decoder log scales.
        decoder_log_mixt_probs (nn.Linear): The linear layer for the decoder log mixture probabilities.
        decoder_binary_stop (nn.Linear): The linear layer for the decoder binary stop.

    Methods:
        forward(data): Performs a forward pass through the model.
        decode(data, latents): Decodes the latent variables into actions.
        prior_inference(data): Performs prior inference.
        encode(data): Encodes the input data.
        _create_decoder_seq(data, latents): Creates the decoder sequence.
        _build_obs_encoder(data): Builds the observation encoder.
        encode_goals(data): Encodes the goals.
        sample_latents(mean, cov): Samples the latent variables.
        compute_prior(data): Computes the prior distribution.
        get_prior(goal_emb): Gets the prior distribution based on the goal embedding.
    """
    def __init__(self, config:dict):
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

    def forward(self, data:dict) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Performs the forward pass of the model.

        Args:
            data: The input data.

        Returns:
            actions: The decoded actions.
            variational_dist: Tuple containing the mean and covariance of the variational distribution.
            prior_dist: Tuple containing the mean and covariance of the prior distribution.
        """
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

    def forward(self, data:dict) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Performs the forward pass of the model.

        Args:
            data: The input data.

        Returns:
            actions: The decoded actions.
            variational_dist: Tuple containing the mean and covariance of the variational distribution.
            prior_dist: Tuple containing the mean and covariance of the prior distribution.
        """
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

    def prior_inference(self, data:dict) -> torch.Tensor:
        """
        Perform prior inference to generate actions based on the given data.

        Args:
            data: The input data for prior inference.

        Returns:
            actions: The actions generated by the prior inference.
        """
        self._build_obs_encoder(data)
        prior_mean, prior_cov = self.compute_prior(data)
        samples = self.sample_latents(prior_mean, prior_cov)
        actions = self.decode(data, samples)
        return actions

    def encode(self, data:dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the given data into a latent representation.

        Args:
            data (dict): The input data dictionary containing the trajectory information.

        Returns:
            tuple: A tuple containing the mean and covariance of the latent representation.
        """
        trajectory = self._build_obs_encoder(data)
        encoding = self.trajectory_encoder(trajectory, data["obs_length"])
        encoding = self.encoder_output(encoding)
        mean, cov = encoding[:, : self.latent_dim], F.softplus(
            encoding[:, self.latent_dim :]
        )
        cov = get_cov(cov)
        return mean, cov
    
    def decode(self, data:dict, latents:torch.Tensor)->torch.Tensor:
        """
        Decodes the given data and latents to generate actions.

        Args:
            data (dict): The input data.
            latents (Tensor): The latent variables.

        Returns:
            tuple: A tuple containing the actions generated by the decoder.
        """
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

    def _create_decoder_seq(self, data:dict, latents:torch.Tensor) -> torch.Tensor:
        """
        Create a decoder sequence by concatenating observations, latents, and goals.

        Args:
            data (dict): A dictionary containing the encoder observations and goals.
            latents (torch.Tensor): The latents tensor.

        Returns:
            torch.Tensor: The concatenated trajectory tensor.
        """
        _, T, _ = data["encoder_obs"].shape
        obs = data["encoder_obs"]
        latents = latents.unsqueeze(1).repeat(1, T, 1)
        goals = data["goals"].unsqueeze(1).repeat(1, T, 1)
        trajectory = torch.cat([obs, latents, goals], dim=-1)

        return trajectory

    def _build_obs_encoder(self, data:dict) -> torch.Tensor:
        """
        Builds the observation encoder for the given data.

        Args:
            data (dict): A dictionary containing the input data.

        Returns:
            torch.Tensor: The encoded observations.

        """
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

    def encode_goals(self, data:dict) -> torch.Tensor:
        """
        Encodes the goals in the given data.

        Args:
            data (dict): A dictionary containing the following keys:
                - "encoder_obs" (torch.Tensor): The encoder observations.
                - "mask" (torch.Tensor): The mask indicating which goals to encode.

        Returns:
            torch.Tensor: The encoded goal embeddings.
        """

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

    def sample_latents(self, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """
        Samples latent variables from a multivariate normal distribution.

        Args:
            mean (torch.Tensor): Mean of the multivariate normal distribution.
            cov (torch.Tensor): Covariance matrix of the multivariate normal distribution.

        Returns:
            torch.Tensor: Sampled latent variables.
        """
        mvn = MultivariateNormal(mean, cov)
        samples = mvn.rsample()
        return samples

    def compute_prior(self, data:dict)->Tuple[torch.Tensor, torch.Tensor]:

        if self.dataset == "calvin":
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

    def get_prior(self, goal_emb:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the prior distribution parameters for the given goal embedding.

        Args:
            goal_emb (torch.Tensor): The goal embedding.

        Returns:
            tuple: A tuple containing the mean and covariance of the prior distribution.
        """
        latent_embeddings = self.prior(goal_emb)
        mean, cov = latent_embeddings[:, : self.latent_dim], F.softplus(
            latent_embeddings[:, self.latent_dim :]
        )
        cov = get_cov(cov)
        return mean, cov


def get_mlp(
    input_dimension: int, hidden_dimension: int, output_dimension: int, n_layers: int
) -> nn.Module:
    """
    Create a multi-layer perceptron (MLP) neural network.

    Args:
        input_dimension (int): The dimension of the input data.
        hidden_dimension (int): The dimension of the hidden layers.
        output_dimension (int): The dimension of the output data.
        n_layers (int): The number of hidden layers in the MLP.

    Returns:
        nn.Module: The MLP neural network.

    Raises:
        AssertionError: If the number of layers is not a positive integer.
    """
    assert n_layers > 0, "Number of layers must be a positive integer"
    layers = [nn.Linear(input_dimension, hidden_dimension), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dimension, hidden_dimension), nn.ReLU()]
    layers.append(nn.Linear(hidden_dimension, output_dimension))
    return nn.Sequential(*layers)


def get_device(config: dict) -> torch.device:
    """
    Get the device to be used for training based on the configuration.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        torch.device: The device to be used for training.
    """
    device = config["device"] if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def get_positional_embeddings(num_embeddings: int, n: int, d: int)->nn.Module:
    """
    Get positional embeddings for a given number of embeddings, sequence length, and embedding dimension.

    Args:
        num_embeddings (int): The number of embeddings.
        n (int): The sequence length.
        d (int): The embedding dimension.

    Returns:
        torch.nn.Embedding: The positional embeddings.
    """
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


def get_n_parameters(model: nn.Module) -> int:
    """
    Calculate the total number of parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        int: The total number of parameters in the model.
    """
    return sum([p.numel() for p in model.parameters()])


def get_tensor_device(tensor: torch.tensor) -> torch.device:
    """
    Returns the device of the given tensor.

    Args:
        tensor (torch.tensor): The input tensor.

    Returns:
        torch.device: The device of the tensor.
    """
    return torch.device(f"cuda:{tensor.get_device()}")


def get_padding_mask(batch_size: int, length: int, seq_length: torch.tensor, device) -> torch.Tensor:
    """
    Generate a padding mask based on the sequence length.

    Args:
        batch_size (int): The size of the batch.
        length (int): The length of the sequence.
        seq_length (torch.tensor): The sequence lengths for each element in the batch.
        device: The device to be used for the computation.

    Returns:
        torch.Tensor: The padding mask.

    """
    padding_mask = torch.arange(0, length).expand(batch_size, length).to(
        device
    ) > seq_length.reshape(-1, 1)
    return padding_mask


def get_sequence_mask(
    T: int, context_length: int, obs_length: torch.tensor, n_head: int, device
) -> torch.Tensor:
    """
    Generate a sequence mask for self-attention mechanism.

    Args:
        T (int): The length of the sequence.
        context_length (int): The length of the context.
        obs_length (torch.tensor): The lengths of the observations.
        n_head (int): The number of attention heads.
        device: The device to be used for computation.

    Returns:
        torch.Tensor: The sequence mask.

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


def get_cov(cov: torch.Tensor)->torch.Tensor:
    """
    Extend the covariance matrix to a square matrix.

    Args:
        cov (torch.Tensor): The input covariance matrix of shape (B, D).

    Returns:
        torch.Tensor: The extended covariance matrix of shape (B, D, D).
    """
    B, D = cov.shape
    extended_cov = torch.zeros(B, D, D).to(cov.device)
    extended_cov[:, torch.arange(D), torch.arange(D)] = cov
    return extended_cov


def log_sum_exp(x:torch.Tensor)->torch.Tensor:
    """
    Numerically stable log_sum_exp implementation that prevents overflow

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: The log-sum-exp of the input tensor along the specified axis.
    """
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))



class FocalLoss(nn.Module):
    """
    Focal Loss implementation.

    Args:
        gamma (float): The focusing parameter. Controls the degree of emphasis on hard examples.

    Attributes:
        gamma (float): The focusing parameter.

    Methods:
        forward(logits, gt): Computes the focal loss given the logits and ground truth labels.

    """

    def __init__(self, gamma: float):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, gt: torch.Tensor)->torch.Tensor:
        """
        Compute the focal loss given the logits and ground truth labels.

        Args:
            logits (torch.Tensor): The predicted logits of shape (BxC), where B is the batch size and C is the number of classes.
            gt (torch.Tensor): The ground truth labels of shape (B), where B is the batch size.

        Returns:
            torch.Tensor: The computed focal loss.

        """
        prob = F.softmax(logits, dim=-1)
        prob = torch.gather(prob, dim=1, index=gt)
        return -((1 - prob) ** self.gamma) * prob.log()


class GripperEncoderCalvin(nn.Module):
    """
    GripperEncoderCalvin is a neural network module that encodes the input image
    using convolutional layers and fully connected layers.

    Args:
        depth (int): The depth of the input image.

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer.
        conv2 (nn.Conv2d): The second convolutional layer.
        conv3 (nn.Conv2d): The third convolutional layer.
        last_conv_output_size (int): The output size of the last convolutional layer.
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
    """

    def __init__(self, depth:int):
        super().__init__()

        self.conv1 = nn.Conv2d(3 + depth, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute the output size of the last convolutional layer
        self.last_conv_output_size = 64 * 7 * 7

        self.fc1 = nn.Linear(self.last_conv_output_size, 256)
        self.fc2 = nn.Linear(256, 32)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
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
    """
    Image encoder module for Calvin model.

    Args:
        depth (int): The depth of the input image.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        conv3 (nn.Conv2d): Third convolutional layer.
        last_conv_output_size (int): Output size of the last convolutional layer.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
    """

    def __init__(self, depth:int):
        super().__init__()

        self.conv1 = nn.Conv2d(3 + depth, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute the output size of the last convolutional layer
        self.last_conv_output_size = 64 * 21 * 21

        self.fc1 = nn.Linear(self.last_conv_output_size, 512)
        self.fc2 = nn.Linear(512, 64)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
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
    """
        DownConvBlock is a class that represents a down-convolutional block in a neural network.
        
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int, optional): The size of the convolutional kernel. Defaults to 3.
            stride (int, optional): The stride of the convolution. Defaults to 1.
            padding (int, optional): The padding of the convolution. Defaults to 1.
            group_size (int, optional): The group size for group normalization. Defaults to 32.
            activation (str, optional): The activation function to use. Defaults to "relu".
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        group_size: int = 32,
        activation: str = "relu",
    ) -> None:
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

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Forward pass of the DownConvBlock.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The output tensor after applying the convolution, normalization, activation, and downsampling.
        """
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
        """
        Positional Encoding module for Transformer models.

        Args:
            d_model (int): The dimension of the input embeddings.
            dropout (float, optional): The dropout probability. Default is 0.1.
            max_len (int, optional): The maximum length of the input sequence. Default is 5000.
        """
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
        Forward pass of the Positional Encoding module.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embedding_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, embedding_dim].
        """
        x = x + self.pe[: x.size(1)].unsqueeze(0)
        return self.dropout(x)


class LayerNorm(nn.Module):
    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        """
        Layer normalization module.

        Args:
            num_channels (int): Number of channels in the input tensor.
            eps (float, optional): Small value added to the denominator for numerical stability. Default is 1e-5.
            affine (bool, optional): If True, apply an affine transformation to the normalized tensor. Default is True.
            device (torch.device, optional): Device on which to create the parameters. Default is None.
            dtype (torch.dtype, optional): Data type of the parameters. Default is None.
        """
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

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Forward pass of the normalization layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, sequence_length).

        Returns:
            torch.Tensor: Normalized output tensor of the same shape as the input.
        """
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


class TrajectoryTransformer(nn.Module):
    """
    A transformer-based model for processing trajectory data.

    Args:
        input_dim (int): The dimension of the input trajectory.
        d_dim (int): The dimension of the transformer's hidden states.
        n_head (int): The number of attention heads in the transformer.
        hidden_dim (int): The dimension of the feedforward network in the transformer.
        num_layers (int): The number of transformer layers.
        dropout (float): The dropout rate.
        aggregate (bool): Whether to aggregate the output trajectory.
        mask_type (str): The type of mask to apply during the transformer computation.
        context_length (int): The length of the context window.
        use_positional_encoding (bool): Whether to use positional encoding in the transformer.

    Attributes:
        aggregate (bool): Whether to aggregate the output trajectory.
        mask_type (str): The type of mask to apply during the transformer computation.
        use_positional_encoding (bool): Whether to use positional encoding in the transformer.
        context_length (int): The length of the context window.
        input_projection (nn.Linear): Linear layer for projecting the input trajectory.
        projection (nn.Linear): Linear layer for projecting the input trajectory.
        positional_encoding (PositionalEncoding): Positional encoding layer for the transformer.
        transformer_layer (nn.TransformerEncoderLayer): Transformer layer.
        transformer_encoder (nn.TransformerEncoder): Transformer encoder.
        n_head (int): The number of attention heads in the transformer.
    """

    def __init__(
        self,
        input_dim: int,
        d_dim: int,
        n_head: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        aggregate: bool,
        mask_type: str,
        context_length: int,
        use_positional_encoding: bool,
    ) -> None:
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

    def forward(self, trajectory:torch.Tensor, obs_length:torch.Tensor)->torch.Tensor:
        """
        Forward pass of the TrajectoryTransformer.

        Args:
            trajectory (torch.Tensor): The input trajectory of shape (B, T, input_dim).
            obs_length (torch.Tensor): The observed length of each trajectory in the batch of shape (B,).

        Returns:
            torch.Tensor: The output trajectory of shape (B, T, d_dim).

        """
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
    
def get_activation(fn_name: str)->nn.Module:
    """
    Returns an activation function based on the given function name.

    Args:
        fn_name (str): The name of the activation function.

    Returns:
        torch.nn.Module: An instance of the requested activation function.

    Raises:
        NotImplementedError: If the requested activation function is not implemented.
    """
    if fn_name == "relu":
        return nn.ReLU()
    else:
        raise NotImplementedError(f"Activation function {fn_name} not implemented")