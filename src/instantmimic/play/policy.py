"""Public playback policy helpers shared by analysis and render tools."""

from __future__ import annotations

import torch

from instantmimic.rl.agent_protocol import PlaybackPolicyAgent


def policy_mean_into(
    agent: PlaybackPolicyAgent,
    obs_policy: torch.Tensor,
    out_actions: torch.Tensor,
) -> None:
    agent.policy_mean_into(obs_policy, out_actions)


@torch.no_grad()
def policy_execution_mean_into(
    agent: PlaybackPolicyAgent, obs_policy: torch.Tensor, out_actions: torch.Tensor
) -> None:
    agent.policy_execution_mean_into(
        obs_policy,
        out_actions,
    )
