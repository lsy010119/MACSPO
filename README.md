# Collision-Free Velocity Scheduling for Multi-Agent Systems on Predefined Routes via Inexact-Projection ADMM

**Anonymous Authors** (under double-blind review)

## Abstract
In structured multi-agent transportation systems, agents often must follow predefined routes, making spatial rerouting undesirable or impossible. This paper addresses route-constrained multi-agent coordination by optimizing waypoint passage times while preserving each agent's assigned route. A differentiable surrogate trajectory model maps waypoint timings to smooth position profiles and captures first-order tracking lag, enabling pairwise safety to be encoded through distance-based penalties evaluated over a dense temporal grid spanning the mission horizon. The resulting nonlinear and nonconvex velocity-scheduling problem is solved using an inexact-projection Alternating Direction Method of Multipliers (ADMM) algorithm that combines structured timing updates with gradient-based collision-correction steps and avoids explicit integer sequencing variables. Numerical experiments on random-crossing, bottleneck, and graph-based network scenarios show that the proposed method computes feasible and time-efficient schedules across a range of congestion levels and yields shorter mission completion times than a representative hierarchical baseline in the tested bottleneck cases.

## Methodology Overview

![](https://anonymous.4open.science/api/repo/MACSPOS-8B07/file/figs/trajex.png)

Our framework coordinates multiple agents on fixed paths by optimizing waypoint passage times.

## Numerical examples

### Case 1: Random Crossing
| Time | Velocity |
|---|---|
| ![](https://anonymous.4open.science/api/repo/MACSPOS-8B07/file/figs/case1_timetraj.png) | ![](https://anonymous.4open.science/api/repo/MACSPOS-8B07/file/figs/case1_veltraj.png) |

![](https://anonymous.4open.science/api/repo/MACSPOS-8B07/file/figs/case1.gif)

### Case 2: Bottleneck
| Time | Velocity |
|---|---|
| ![](https://anonymous.4open.science/api/repo/MACSPOS-8B07/file/figs/case2_timetraj.png) | ![](https://anonymous.4open.science/api/repo/MACSPOS-8B07/file/figs/case2_veltraj.png) |

![](https://anonymous.4open.science/api/repo/MACSPOS-8B07/file/figs/case2.gif)

### Case 3: Graph Trajectory
| Time | Velocity |
|---|---|
| ![](https://anonymous.4open.science/api/repo/MACSPOS-8B07/file/figs/case3_timetraj.png) | ![](https://anonymous.4open.science/api/repo/MACSPOS-8B07/file/figs/case3_veltraj.png) |

| Demo | With stops |
|---|---|
| ![](https://anonymous.4open.science/api/repo/MACSPOS-8B07/file/figs/case3.gif) | ![](https://anonymous.4open.science/api/repo/MACSPOS-8B07/file/figs/case3t.gif) |

The results shown on the right demonstrate the framework's capability in scenarios requiring mission-specific stops, such as Urban Air Mobility (UAM) or Automated Logistics.
