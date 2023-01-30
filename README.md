
# 🍌 Stable Diffusion 2.1 on Banana

This repo gives a basic framework for serving Stable Diffusion 2.1 in production using simple HTTP servers.

# Quickstart

Deploy this model via the one-click template [here]((https://app.banana.dev/templates/djt/stable-diffusion-2.1).

You will also find the model inputs and outputs to help get you going.

Code snippets are visible at the bottom (more coming soon).

### Additional Steps 

1. Create huggingface account to get permission to download and run [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) text-to-image model.
  - Accept terms and conditions for the use of [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)
2. After deploying the model on Banana, be sure to set the HF_AUTH_TOKEN build argument in the settings page of the model on Banana

# Helpful Links
Understand the 🍌 [Serverless framework](https://docs.banana.dev/banana-docs/core-concepts/inference-server/serverless-framework) and functionality of each file within it.

Generalize this framework to [deploy anything on Banana](https://docs.banana.dev/banana-docs/resources/how-to-serve-anything-on-banana).

## Use Banana for scale.
