<p align="center">

  <h2 align="center">Collision-free Velocity Profile Optimization via First-Order Method With Inexact Projection</h2>
  <p align="center">
    <a><strong>Seungyeop Lee</strong></a><sup>1</sup>
    ·
    <a><strong>Jong-Han Kim</strong></a><sup>1</sup>
</p>

<p align="center">
    <sup>1</sup>Department of Aerospace Engineering, Inha University, Incheon, 21999, Republic of Korea
</p>
   <h3 align="center">
  <div align="center"></div>
</p>

## Abstract
In cooperative multi-agent missions, the design of collision-avoidance trajectories is essential. Numerous studies have proposed methods that generate avoidance trajectories satisfying inter-agent safety distances, typically resulting in modified trajectories deviating from the nominal path. However, for systems that must strictly follow a prescribed path or cannot significantly deviate from the reference trajectory, such approaches are only applicable to a limited extent. In contrast, an alternative approach that avoids collisions by adjusting only the velocity along the given path, rather than redesigning the trajectory, is free from such restrictions. In this paper, we formulate an optimization problem with waypoint arrival times as decision variables and solve it using the Alternating Direction Method of Multipliers (ADMM) to generate an optimal collision-free velocity profile.

## Methodology Overview

<p align="center">
    <img src="https://github.com/user-attachments/assets/d95378b5-0f19-43a8-91e0-ae8fdfb59f1d" alt="time" width="500"/>
</p>

Our framework focuses on coordinating multiple agents within a pre-defined path network. The methodology is built upon the following key assumptions and formulations:

- **Fixed Path Assignment:** Each agent $i$ is assigned a sequence of waypoints that must be visited in order.

- **Linear-Segment Paths:** The agents are assumed to move along piece-wise linear paths connecting these waypoints. This representation is particularly suitable for structured environments like UAM corridors or warehouse aisles.

- **Constant Segment Velocity:** Between any two consecutive waypoints, the agent is assumed to maintain a constant velocity. This simplifies the high-level coordination while allowing for velocity changes at each waypoint.

Based on these assumptions, the passage times at each waypoint are defined as the primary decision variables. By optimizing these times, the algorithm yields a velocity profile that ensures safety without deviating from the assigned spatial path. Based on the assumptions, we define the coordination problem as an optimization of passage times $\{\mathbf{t}^{(i)}\}$ to minimize the total mission time while satisfying kinematic and safety constraints:

```math
\begin{aligned}
    \underset{\{\mathbf{t}^{(i)}\}_{i\in[K]}}{\rm min}\;\; 
    & \sum_{i\in [K]\setminus\mathcal{K}_f^c} t^{(i)}_{N_i} \\    
    \text{s.t.}\;\;                              
    & t^{(i)}_1 = t^{(i)}_{s}, \; \forall i\in[K]\\        
    & t^{(i)}_{N_i} = t^{(i)}_f, \; \forall i\in \mathcal{K}_f\\
    & \frac{{\bf d}^{(i)}}{v_{\max}} \leq \Delta {\bf t}^{(i)} \leq \frac{{\bf d}^{(i)}}{v_{\min}}, \; \forall i\in[K]\\        
    & \|{\bf p}^{(i)}({\bf t}^{(i)},\tau)-{\bf p}^{(j)}({\bf t}^{(j)},\tau)\|_2\geq d_{\rm safe}, \\
    & \forall (i,j) \in \mathcal{P}, \; \forall \tau \in \mathcal{T}.       
\end{aligned}
```

To ensure the trajectories are $C^1$-continuous and physically feasible, we utilize a sigmoid-based approximation model for the agent's position $\mathbf{p}^{(i)}(\tau)$:

```math
\begin{aligned}
{\bf p}^{(i)}({\bf t}^{(i)},\tau) &=\int^\tau_0 {\bf v}^{(i)}({\bf t}^{(i)},\tau){\rm d}t \\
&= {\bf p}^{(i)}_1 + \sum^{N_i-1}_{n=1}\frac{{\bf v}^{(i)}_n({\bf t}^{(i)})}{\beta}\Big\{ \zeta(\beta (\tau-\hat{t}^{(i)}_n)) \\
&\quad -\zeta(\beta (\tau-\hat{t}^{(i)}_{n+1}))\Big\}.
\end{aligned}
```

The resulting non-convex, non-linear optimization problem is solved using the Alternating Direction Method of Multipliers (ADMM) framework. This allows for efficient first-order updates, making it scalable for large-scale multi-agent systems.


## Numerical examples
### Case 1: Random Crossing
High-density agents intersecting at random angles.

<p align="center">
    <img src="https://github.com/user-attachments/assets/d95378b5-0f19-43a8-91e0-ae8fdfb59f1d" alt="time" width="500"/>
    <img src="https://github.com/user-attachments/assets/cb2f982b-66a3-435a-ab9d-6c4d810b8c1b" alt="speed" width="500"/>
</p>

### Case 2: Bottleneck
Agents converging into a narrow passage.

### Case 3: Graph Trajectory
Large-scale coordination on a complex network (UAM/Logistics).
