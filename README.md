
<div align="center">

# Helios-B200-Unleashed: Real-Time High-Quality Video Generation

</div>

## Introduction

Helios is a groundbreaking 14B video generation model designed for real-time performance and robustness to long-video drifting. While currently demonstrating impressive capabilities on NVIDIA H100 GPUs, this project, **Helios-B200-Unleashed**, outlines our strategic roadmap to push the boundaries further. Our ambition is to achieve a sustained **24 Frames Per Second (FPS) with unparalleled visual quality** on the next-generation NVIDIA B200 GPU, leveraging advanced quantisation and reinforcement learning techniques.

## Current Benchmarks: Helios on NVIDIA H100

The Helios model currently delivers a remarkable **19.5 FPS** on a single NVIDIA H100 GPU for minute-scale video generation. This achievement is notable as it does not rely on standard acceleration techniques like KV-cache, sparse/linear attention, or quantization, highlighting Helios's inherent efficiency. It proficiently handles Text-to-Video (T2V), Image-to-Video (I2V), and Video-to-Video (V2V) tasks.

## The Power of NVIDIA B200: A New Era for AI Inference

The NVIDIA B200 GPU, powered by the innovative Blackwell architecture, represents a monumental leap in AI inference capabilities. Benchmarking indicates significant advantages in throughput and cost-effectiveness compared to its predecessors (H100 and H200). The B200 delivers up to **15x performance improvements** over the Hopper generation, primarily due to:

*   **NVFP4 Precision:** A new floating-point format designed for extreme efficiency.
*   **NVLink 5 Interconnects:** Enhanced bandwidth for multi-GPU communication.
*   **TensorRT-LLM & Dynamo:** Software optimizations accelerating large language model inference, which can be adapted for video generation.

These innovations provide the foundational hardware capabilities essential for our 24 FPS target.

## Future Roadmap for 24 FPS with High Visual Quality on B200

To transcend the current performance and achieve 24 FPS with high visual fidelity on the B200, our roadmap focuses on two critical integration pathways: LightX2V and GenRL.

### LightX2V Integration for Extreme Performance (FP8/NVFP4 Quantisation)

LightX2V is an advanced inference framework specifically engineered for light image and video generation. Its integration will be pivotal in optimizing Helios's execution on the B200:

*   **FP8/NVFP4 Quantisation:** We will leverage LightX2V's support for FP8 (and B200's native NVFP4) per-tensor quantisation. This will drastically reduce the memory footprint and computational requirements, enabling higher throughput without significant quality degradation. LightX2V has demonstrated accelerations of approximately **42x** with FP8 models and 4-step CFG/step distillation on H100, providing a strong basis for even greater gains on B200.
*   **CFG Parallelism and Block-Level Offload:** Further optimizations from LightX2V, such as CFG parallelism and intelligent block-level offloading, will ensure efficient resource utilization and minimize latency, directly contributing to our FPS goal.

### GenRL for Advanced Motion and Aesthetic Tuning (FlowGRPO Reinforcement Learning)

While LightX2V focuses on raw performance, GenRL will be instrumental in elevating the visual quality and coherence of the generated videos:

*   **FlowGRPO Reinforcement Learning:** We will implement FlowGRPO reinforcement learning, a core component of GenRL, for sophisticated motion and aesthetic tuning. This technique enables multi-reward optimization, using metrics like HPSv3 and VideoAlign, to refine the diffusion and flow models within Helios.
*   **Enhanced Motion Coherence & Aesthetic Quality:** By applying FlowGRPO, we anticipate significant improvements in video output, specifically in achieving fluid motion, reducing artifacts, and boosting overall aesthetic appeal to meet professional quality standards. GenRL's high-performance LoRA checkpoints provide a strong foundation for these enhancements.

## Achieving the 24 FPS & High Quality Synergy

The combination of the NVIDIA B200's raw computational power, LightX2V's aggressive inference optimizations (especially FP8/NVFP4 quantisation), and GenRL's sophisticated quality enhancements (FlowGRPO reinforcement learning) creates a synergistic approach. This trifecta will enable Helios to not only surpass the 24 FPS benchmark but also set a new standard for real-time, high-visual-quality video generation. We are confident this roadmap will unleash the full potential of Helios on the B200 architecture.

## Conclusion & Next Steps

Helios-B200-Unleashed represents our commitment to leading the frontier of real-time video generation. With the strategic integration of LightX2V and GenRL on the NVIDIA B200 platform, we are poised to deliver unprecedented performance and visual fidelity. We look forward to sharing our progress and the continued evolution of this exciting project.
