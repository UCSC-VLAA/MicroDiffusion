# MicroDiffusion: mplicit Representation-Guided Diffusion for 3D Reconstruction from Limited 2D Microscopy Projections

This is the official repository for our paper: "MicroDiffusion: mplicit Representation-Guided Diffusion for 3D Reconstruction from Limited 2D Microscopy Projections."


## Abstract:
Volumetric optical microscopy using non-diffracting beams enables rapid imaging of 3D volumes by projecting them axially to 2D images but lacks crucial depth information. Addressing this, we introduce MicroDiffusion, a pioneering tool facilitating high-quality, depth-resolved 3D volume reconstruction from limited 2D projections. 
While existing Implicit Neural Representation (INR) models often yield incomplete outputs and Denoising Diffusion Probabilistic Models (DDPM) excel at capturing details, our method integrates INR's structural coherence with DDPM's fine-detail enhancement capabilities. We pretrain an INR model to transform 2D axially-projected images into a preliminary 3D volume. This pretrained INR acts as a global prior guiding DDPM's generative process through a linear interpolation between INR outputs and noise inputs. This strategy enriches the diffusion process with structured 3D information, enhancing detail and reducing noise in localized 2D images.
By conditioning the diffusion model on the closest 2D projection, MicroDiffusion substantially enhances fidelity in resulting 3D reconstructions, surpassing INR and standard DDPM outputs with unparalleled image quality and structural fidelity.


<div align="center">
  <img src="figures/model.png"/>
</div>
<div align="center">
  <img src="figures/panel.png"/>
</div>

## Acknowledge
This work is partially supported by TPU Research Cloud (TRC) program, and Google Cloud Research Credits program.


## Coming Soon!
We are preparing the code, data and models for release. Stay tuned!
