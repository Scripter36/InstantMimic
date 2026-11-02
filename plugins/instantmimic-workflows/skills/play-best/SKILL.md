---
name: play-best
description: Use when evaluating a checkpoint, finding a best checkpoint, generating playback video, or extracting playback reward and episode length.
---

# Play Best

Post-training evaluation uses `play` from the best checkpoint.

## Checkpoint Resolution

Prefer explicit checkpoint path when known:

```bash
++play.checkpoint=<run_dir>/checkpoints/best_agent.pt
```

An explicit checkpoint restores its saved configuration, then applies explicit
CLI dotlist overrides. An alternate `--config-name` does not replace those saved
settings. Set evaluation reset, noise, pushes, motion sampling and horizon as
explicit overrides when needed; verify the effective protocol before reporting.

When omitted, the play resolver selects `best_agent.pt` from the newest matching
run that contains one. Explicit values must name an existing checkpoint file.

The helper below selects by file modification time, not by comparing quality
across runs. Use an explicit run/checkpoint for reproducible evaluation.

Helper:

```bash
uv run python plugins/instantmimic-workflows/scripts/find_best_checkpoint.py [path-or-pattern]
```

## Reportable Video

Read [Optional Local Workflows](../../references/optional-local-workflows.md) and follow the active repository or host GPU-coordination policy before local playback/video commands.

```bash
uv run play play.video.enabled=true ++play.video.overlay_rewards=true ++play.video.overlay_history_seconds=3.0 ++play.video.overlay_panel_width=360 ++play.checkpoint=<ckpt_path>
```

## Record

- checkpoint path
- video path from `Saved video to:`
- `PLAYBACK_RESULT_JSON` reward, episode length and protocol fields (prefer the
  structured line; legacy `PLAYBACK_RESULT` is available)
- effective evaluation settings and complete-episode/termination behavior
- for video: ffprobe metadata and inspected beginning/middle/end frames
