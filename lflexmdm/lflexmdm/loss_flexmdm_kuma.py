from typing import Optional, cast

from jaxtyping import Float
import torch
import torch.nn.functional as F
from torch import Tensor as TT
from .types_flexmdm import (
    FlexMDMAuxModel,
    FlexMDMBatch,
    FlexMDMLossDict,
    FlexMDMModel,
)
from xlm.harness import LossFunction, Harness
from xlm.datamodule import Tokenizer
from .schedules import (
    FlexMDMSchedule,
    Schedule,
    log1mexp_exact_safegrad as log1mexp,
)
from .utils import bregman_divergence
import xlm.flags as flags

if hasattr(flags, "DEBUG_NO_PHI_LOSS"):
    DEBUG_NO_PHI_LOSS = flags.DEBUG_NO_PHI_LOSS
else:
    DEBUG_NO_PHI_LOSS = False

if hasattr(flags, "DEBUG_LOG_PHI"):
    DEBUG_LOG_PHI = flags.DEBUG_LOG_PHI
else:
    DEBUG_LOG_PHI = False


def sample_time(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Stratified sampling of time t from Uniform[0, 1-eps] with batch_size strata.
    """
    eps = 1e-6
    interval = 1.0 - eps
    interval_size = interval / batch_size
    u = torch.rand(batch_size, device=device)
    return (
        torch.arange(batch_size, device=device, dtype=u.dtype) + u
    ) * interval_size


class LFlexMDMLoss(LossFunction[FlexMDMBatch, FlexMDMLossDict]):
    """Learned noise FlexMDM loss with REINFORCE gradient estimation."""

    model: FlexMDMModel
    aux_model: FlexMDMAuxModel

    def __init__(
        self,
        noise_schedule: FlexMDMSchedule,
        model: Optional[FlexMDMModel] = None,
        aux_model=None,  # FlexMDMAuxModel
        tokenizer: Optional[Tokenizer] = None,
        eps: float = 1e-6,
        regularizer_weight: float = 0.0,
        phi_loss: bool = True,
        max_length: float = 1.0,
        stop_grad_on_phi: bool = False,
        use_length_scale: bool = True,
        use_unmask_counts: bool = True,
        reinforce_weight: float = 1.0,
        _use_t: bool = True,  # for a future model, use False
    ):
        self.noise_schedule = noise_schedule
        self.model = model
        self.aux_model = aux_model
        self.tokenizer = tokenizer
        self.mask_token_id_tensor = None
        self.pad_token_id_tensor = None
        self.eps = eps
        self.regularizer_weight = regularizer_weight
        self.phi_loss = phi_loss
        self.max_length = max_length
        self.stop_grad_on_phi = stop_grad_on_phi
        self.use_length_scale = use_length_scale
        self.use_unmask_counts = use_unmask_counts
        self.reinforce_weight = reinforce_weight
        self._use_t = _use_t

    def configure(self, pl_module: Harness):
        self.mask_token_id_tensor = torch.tensor(
            self.tokenizer.mask_token_id,
            dtype=torch.long,
            device=pl_module.device,
        )
        self.pad_token_id_tensor = torch.tensor(
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=pl_module.device,
        )

    def __call__(
        self,
        batch: FlexMDMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> FlexMDMLossDict:
        return self.loss_fn(batch, batch_idx, dataloader_idx, dataloader_name)

    def loss_fn(
        self,
        batch: FlexMDMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> FlexMDMLossDict:
        if self.phi_loss:
            return self._loss_fn(
                batch, batch_idx, dataloader_idx, dataloader_name
            )
        else:
            return self._loss_fn_no_phi(
                batch, batch_idx, dataloader_idx, dataloader_name
            )

    def _loss_fn_no_phi(
        self,
        batch: FlexMDMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> FlexMDMLossDict:
        # Get clean sequences
        z_1 = batch["input_ids"]  # (B, L)
        fixed = batch["fixed"]

        batch_size, max_seq_len = z_1.shape
        device = z_1.device

        # Step 1: Sample time t ~ Uniform(0,1) per sequence
        # t = (1 - self.eps) * t  # ensure t is not exactly 1
        t = self.noise_schedule.sample_t(
            (batch_size,), device, antithetic=True
        )

        # Step 3: Forward pass aux model to get a^{φ,i}(z_1) and b^{φ,i}(z_1)
        attention_mask = (z_1 != self.pad_token_id_tensor).bool()
        t_expaned = t.unsqueeze(-1).expand_as(z_1)
        hazard_ins = self.noise_schedule.insertion_hazard_rate(t_expaned, None)
        hazard_unmask = self.noise_schedule.unmasking_hazard_rate(
            t_expaned, None
        )
        # Step 4: Sample TWO variable-length masked versions
        # Sample insertion times: T_ins ~ Exp(a_phi * alpha_t)
        t_ins_1, t_unmask_1 = self.noise_schedule.sample_ins_unmask_times(
            t_expaned, None
        )

        x_t_1, s_t_1, gaps_1, gaps_mask_1, gap_sums_1, deleted_1, masked_1 = (
            self.noise_schedule.sample_varlen_masked_sequence(
                z_1,
                t_ins_1,
                t_unmask_1,
                t,
                self.mask_token_id_tensor.item(),
                self.pad_token_id_tensor.item(),
                fixed.logical_or(~attention_mask),
                hazard_ins,
            )
        )

        # Step 5: Forward pass main model for both samples
        # Can do a single forward pass by expanding batch dimension
        attention_mask_1 = (x_t_1 != self.pad_token_id_tensor).bool()

        assert self.model is not None
        params_theta_1 = self.model(x_t_1, t, attention_mask_1)

        hazard_ins_theta_1 = self.noise_schedule.insertion_hazard_rate(
            t.unsqueeze(-1), params_theta_1
        )
        hazard_unmask_theta_1 = self.noise_schedule.unmasking_hazard_rate(
            t.unsqueeze(-1), params_theta_1
        )

        loss_1, unmask_loss_1, insertion_loss_1 = (
            self.noise_schedule.compute_generator_loss(
                z_1,
                x_t_1,
                s_t_1,
                gaps_mask_1,
                gap_sums_1,
                params_theta_1["vocab_logits"],
                [hazard_ins, hazard_unmask],  # phi
                [hazard_ins_theta_1, hazard_unmask_theta_1],
                self.mask_token_id_tensor,
                lenght_scale=self.max_length if self.use_length_scale else 1.0,
                use_unmask_counts=self.use_unmask_counts,
            )
        )

        total_loss = loss_1
        params_theta_debug = None
        if DEBUG_LOG_PHI:
            # send params_phi only for the relevant positions
            params_theta_debug = {}
            for name, value in params_theta_1.items():
                if value is not None:
                    value_debug = value.clone().detach()
                    value_debug[~attention_mask_1.bool()] = torch.nan
                    params_theta_debug[name] = value_debug
                else:
                    params_theta_debug[name] = None
        return {
            "loss": total_loss.mean(),
            "unmask_loss": (
                unmask_loss_1.detach().mean()
            ).detach(),  # approximate split
            "insertion_loss": insertion_loss_1.detach().mean(),
            "reg_loss": torch.zeros_like(total_loss).detach(),
            "theta_params": params_theta_debug,
        }

    def _loss_fn(
        self,
        batch: FlexMDMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> FlexMDMLossDict:
        # Get clean sequences
        z_1 = batch["input_ids"]  # (B, L)
        fixed = batch["fixed"]

        batch_size, max_seq_len = z_1.shape
        device = z_1.device

        # Step 1: Sample time t ~ Uniform(0,1) per sequence
        # t = torch.rand(batch_size, device=device)  # (B,)
        t = self.noise_schedule.sample_t(
            (batch_size,), device, antithetic=True
        )

        # t = (1 - self.eps) * t  # ensure t is not exactly 1

        # Step 3: Forward pass aux model to get a^{φ,i}(z_1) and b^{φ,i}(z_1)
        attention_mask = (z_1 != self.pad_token_id_tensor).bool()
        if DEBUG_NO_PHI_LOSS or not self.phi_loss:
            temp_ = torch.ones_like(z_1).to(dtype=t.dtype)
            params_phi = {
                "b_ins": temp_,
                "b_unmask": temp_,
                "a_ins": temp_,
                "a_unmask": temp_,
            }
        else:
            params_phi = self.aux_model(
                z_1,
                (t if self._use_t else torch.ones_like(t)),
                attention_mask,
            )  # dict with b_ins, b_unmask, and optionally a_ins, a_unmask
        hazard_ins = self.noise_schedule.insertion_hazard_rate(
            t.unsqueeze(-1), params_phi
        )
        hazard_unmask = self.noise_schedule.unmasking_hazard_rate(
            t.unsqueeze(-1), params_phi
        )
        # Step 4: Sample TWO variable-length masked versions
        # Sample insertion times: T_ins ~ Exp(a_phi * alpha_t)
        t_ins_1, t_unmask_1 = self.noise_schedule.sample_ins_unmask_times(
            t, params_phi
        )

        x_t_1, s_t_1, gaps_1, gaps_mask_1, gap_sums_1, deleted_1, masked_1 = (
            self.noise_schedule.sample_varlen_masked_sequence(
                z_1,
                t_ins_1,
                t_unmask_1,
                t,
                self.mask_token_id_tensor.item(),
                self.pad_token_id_tensor.item(),
                fixed.logical_or(~attention_mask),
                hazard_ins,
            )
        )
        t_ins_2, t_unmask_2 = self.noise_schedule.sample_ins_unmask_times(
            t, params_phi
        )
        x_t_2, s_t_2, gaps_2, gaps_mask_2, gap_sums_2, deleted_2, masked_2 = (
            self.noise_schedule.sample_varlen_masked_sequence(
                z_1,
                t_ins_2,
                t_unmask_2,
                t,
                self.mask_token_id_tensor.item(),
                self.pad_token_id_tensor.item(),
                fixed.logical_or(~attention_mask),
                hazard_ins,
            )
        )

        # Step 5: Forward pass main model for both samples
        # Can do a single forward pass by expanding batch dimension
        attention_mask_1 = (x_t_1 != self.pad_token_id_tensor).bool()
        attention_mask_2 = (x_t_2 != self.pad_token_id_tensor).bool()

        assert self.model is not None
        # TODO: Generalize model signature
        params_theta_1 = self.model(x_t_1, t, attention_mask_1)
        params_theta_2 = self.model(x_t_2, t, attention_mask_2)

        hazard_ins_theta_1 = self.noise_schedule.insertion_hazard_rate(
            t.unsqueeze(-1), params_theta_1
        )
        hazard_unmask_theta_1 = self.noise_schedule.unmasking_hazard_rate(
            t.unsqueeze(-1), params_theta_1
        )
        hazard_ins_theta_2 = self.noise_schedule.insertion_hazard_rate(
            t.unsqueeze(-1), params_theta_2
        )
        hazard_unmask_theta_2 = self.noise_schedule.unmasking_hazard_rate(
            t.unsqueeze(-1), params_theta_2
        )

        return_per_token = False  # Cannot work per token for REINFORCE loss
        loss_1, unmask_loss_1, insertion_loss_1 = (
            self.noise_schedule.compute_generator_loss(
                z_1,
                x_t_1,
                s_t_1,
                gaps_mask_1,
                gap_sums_1,
                params_theta_1["vocab_logits"],
                (
                    [hazard_ins, hazard_unmask]
                    if not self.stop_grad_on_phi
                    else [hazard_ins.detach(), hazard_unmask.detach()]
                ),
                [hazard_ins_theta_1, hazard_unmask_theta_1],
                self.mask_token_id_tensor,
                lenght_scale=self.max_length if self.use_length_scale else 1.0,
                use_unmask_counts=self.use_unmask_counts,
                return_per_token=return_per_token,
            )
        )
        loss_2, unmask_loss_2, insertion_loss_2 = (
            self.noise_schedule.compute_generator_loss(
                z_1,
                x_t_2,
                s_t_2,
                gaps_mask_2,
                gap_sums_2,
                params_theta_2["vocab_logits"],
                (
                    [hazard_ins, hazard_unmask]
                    if not self.stop_grad_on_phi
                    else [hazard_ins.detach(), hazard_unmask.detach()]
                ),
                [hazard_ins_theta_2, hazard_unmask_theta_2],
                self.mask_token_id_tensor,
                lenght_scale=self.max_length if self.use_length_scale else 1.0,
                use_unmask_counts=self.use_unmask_counts,
                return_per_token=return_per_token,
            )
        )

        # Step 7: REINFORCE with leave-one-out baseline
        loss_theta = 0.5 * (loss_1 + loss_2)  # (B,) or (B, L)

        phi_attention_mask = attention_mask & (~fixed)
        log_p_phi_1 = self.compute_log_prob_phi(
            params_phi,
            t,
            phi_attention_mask,
            deleted_1,
            masked_1,
            return_per_token=return_per_token,
        )  # (B,) or (B, L)
        log_p_phi_2 = self.compute_log_prob_phi(
            params_phi,
            t,
            phi_attention_mask,
            deleted_2,
            masked_2,
            return_per_token=return_per_token,
        )  # (B,) or (B, L)

        # REINFORCE gradient
        loss_phi_reinforce = 0.5 * (
            (loss_1.detach() - loss_2.detach()) * (log_p_phi_1 - log_p_phi_2)
        )  # (B,) or (B, L)

        reg_loss = self.noise_schedule.regularizer(
            t.unsqueeze(-1),
            params_phi,
            mask=phi_attention_mask,  # attention_mask
        )  # (B,)

        # Total loss
        if DEBUG_NO_PHI_LOSS or not self.phi_loss:
            total_loss = loss_theta.mean() if return_per_token else loss_theta
        else:
            total_loss = (
                loss_theta
                + (self.reinforce_weight * loss_phi_reinforce)
                + (self.regularizer_weight * reg_loss)
            )  # (B,)

        params_phi_debug = None
        params_theta_debug = None
        if DEBUG_LOG_PHI:
            # send params_phi only for the relevant positions
            params_phi_debug = {}
            for name, value in params_phi.items():
                if value is not None:
                    value_debug = value.clone().detach()
                    value_debug[~phi_attention_mask.bool()] = torch.nan
                    params_phi_debug[name] = value_debug
                else:
                    params_phi_debug[name] = None
            params_theta_debug = {}
            for name, value in params_theta_1.items():
                if value is not None:
                    value_debug = value.clone().detach()
                    value_debug[~attention_mask_1.bool()] = torch.nan
                    params_theta_debug[name] = value_debug
                else:
                    params_theta_debug[name] = None

        return {
            "loss": total_loss.mean() if not return_per_token else total_loss,
            "loss_theta": loss_theta.detach(),
            "advantage": (loss_1.detach() - loss_2.detach()),
            "log_p_diff": (log_p_phi_1.detach() - log_p_phi_2.detach()),
            "unmask_loss": (
                0.5 * (unmask_loss_1.mean() + unmask_loss_2.mean())
            ).detach(),  # approximate split
            "insertion_loss": (
                0.5 * (insertion_loss_1.mean() + insertion_loss_2.mean())
            ).detach(),  # approximate split
            "reg_loss": reg_loss.detach().mean(),
            "phi_params": params_phi_debug,
            "theta_params": params_theta_debug,
        }

    def compute_log_prob_phi(
        self,
        params_phi: dict,
        t: torch.Tensor,
        attention_mask: torch.Tensor,
        deleted: torch.Tensor,
        masked: torch.Tensor,
        return_per_token: bool = False,
    ) -> torch.Tensor:
        """

        Args:
            b_ins_phi: Per-position insertion parameters b_ins^{φ,i}(z_1) (batch, max_seq_len)
            b_unmask_phi: Per-position unmasking parameters b_unmask^{φ,i}(z_1) (batch, max_seq_len)
            t: Diffusion time in [0,1] (batch,)
            attention_mask: Positions to include in log-prob (e.g. valid & not-fixed) (batch, max_seq_len)
            deleted: Boolean mask for deleted positions in z_1 (batch, max_seq_len)
            masked: Boolean mask for masked positions in z_1 (batch, max_seq_len)

        Returns:
            log_prob_total: Sum of log probabilities over valid positions
        """
        # Delegate the state log-prob computations to the schedule container.
        # NOTE:
        # - For schedule_type="simplified-kuma", param=(b_ins_phi, b_unmask_phi).
        # - For schedule_type="simplified-kuma2", the same tensors are interpreted as (a, b).

        log_prob_deleted = self.noise_schedule.log_likelihood_dropped(
            t, params_phi
        )  # (B, L)
        log_prob_masked = self.noise_schedule.log_likelihood_masked(
            t, params_phi
        )  # (B, L)
        log_prob_original = self.noise_schedule.log_likelihood_unmasked(
            t, params_phi
        )  # (B, L)

        # Select log-probability for each position based on sampled state
        log_p_t = torch.where(
            deleted,
            log_prob_deleted,
            torch.where(masked, log_prob_masked, log_prob_original),
        )

        # Apply attention mask to only include valid positions
        log_p_t = torch.where(
            attention_mask.bool(), log_p_t, torch.zeros_like(log_p_t)
        )

        # Compute masked mean (per-position loss to match scale of loss on theta)
        # Handle empty sequences by clamping the denominator
        # num_valid_positions = attention_mask.sum()
        # log_prob_mean = log_p_t_masked.sum() / torch.clamp(
        #    num_valid_positions, min=1.0
        # )

        if return_per_token:
            return log_p_t
        return log_p_t.sum(-1)  # (B,)
