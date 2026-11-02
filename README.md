# InstantMimic

InstantMimic learns physics-based character skills on the GPU in seconds.

[![Walk, run, spinkick, backflip and cartwheel learned in 1.14, 0.88, 1.24, 1.93 and 1.36 seconds of active training](docs/assets/motion-preview.gif)](https://scripter36.github.io/projects/instantmimic/#motion-demo)

The [project page](https://scripter36.github.io/projects/instantmimic/#motion-demo)
shows these policies with their training times.

Because training is this fast, an LLM agent can plan and run whole
[hyperparameter searches](docs/guides/hpo.md). The repository also includes the
[MUSUBI](docs/guides/muscle-control.md#musubi) muscle-control experiments,
[AMASS](docs/data/amass-retargeting.md) latent policies and a
[BeyondMimic](docs/guides/beyondmimic.md) tracking task for the Unitree G1.

## Getting started

Follow the [installation](docs/getting-started/installation.md) guide, then
train your [first policy](docs/getting-started/first-run.md). The
[documentation index](docs/index.md) lists the other guides.

If you work with a coding agent, you can instead install the
[InstantMimic skill](docs/guides/assistant-workflows.md#start-without-a-checkout)
and describe what you want; the agent sets up the repository for you.

## License

Code is licensed under [Apache-2.0](LICENSE). Models and motions have their own
terms, listed in the [data inventory](data/README.md) and
[third-party notices](THIRD_PARTY_NOTICES.md).
