# Reinforcement Learning Project - Actor-Critic learning to fly
This repository is for solving the normal rotor and tilt-rotor enviroment of quadcopter using actor-critic (A2C and A3C) methods and comparing the performance with DQN and DDQN. The study proposes an A2C with 8 tails for actor and a tail for critic to solve the tiltrotor problem. The propsed approach is basically On-policy advantage actor critic with added entropy loss and advantage.

## Enviroment
 * Observation Space:
   * Vector of 18 for Normal Rotor (3xError in Desired Position, 3xError in Body Rates, 3xError in Desire Velocity, 3x3 flattened Rotation Matrix)
   * Vector of 22 for Tiltrotor (Normal Observation Vector of 18 + 4 Errors in tilt rotor)
* Action Space – Continuous
   * 4 continuous actions for each normal rotor
   * 4 continuous actions for normal and 4 for tiltrotor
   * Fuzzification (0 – 1 for normal rotors, -1 – 1 for tiltrotors with 0.2 step) or vector quantization
 
<img align="center" width="1039" src="https://user-images.githubusercontent.com/55484402/146592754-f888c283-a7f4-4d33-9352-d6225660ac8b.png">

## Reward Function
The reward function is given below where there is a positive reward for staying alive (not crashing) and penalties for errors in action, velocity, position, roll and pitch.


<img align="center" width="1039" src="https://render.githubusercontent.com/render/math?math=r_{t}=\beta-\alpha_{a}\|a\|_{2}-\sum_{k \in\{p, v, \omega\}} \alpha_{k}\left\|e_{k}\right\|_{2}-\sum_{j \in \mid \phi, \theta\}} \alpha_{j}\left\|e_{j}\right\|_{2}-\sum_{j \in \mid \phi, \theta\}} \alpha_{j}\left\|e_{j}\right\|_{2}">

  * β ≥ 0 reward for staying alive
  * 𝛂_∗ = 𝑤𝑒𝑖𝑔𝑡ℎ𝑠 𝑓𝑜𝑟 𝑣𝑎𝑟𝑖𝑜𝑢𝑠 𝑡𝑒𝑟𝑚𝑠
  * ∥𝒂∥_𝟐 = Penalty for wrong actions
  * 𝒆_𝒌  = error in position (ep), velocity (ev), and body rates (eω) 
  * 𝒆_𝒋  = error in roll (eφ) , pitch (eθ) 

## Proposed A2C Architecture for Tilt-Rotor
The following is the proposed A2C architecture to solve the tilt-rotor enviroment. The architecture have three branches, one for normal rotors, one for tilt-rotors and one critic branch each policy network branch have 4 tails (one for each rotor). There are concatination between normal and rotor branches to share knowledge.
![image](https://user-images.githubusercontent.com/55484402/146600555-0ce0aed6-f2d3-441a-b186-eef348b76aa2.png)


## Results
A2C, A3C, DQN and DDQN were compared with the normal rotor quadcopter enviroment where you can turn on each rotor at different level of speeds. A2C and A3C had the same network architecture and DQN and DDQN had the same architecture and parameters. The comparision is made against time.

### Comparision between RL methods on Normal Rotor
![image](https://user-images.githubusercontent.com/55484402/146594183-ee80b8cd-d70e-47b8-b5a7-73897e8e3fcf.png)

### A2C Learning graph for Tilt Rotor
Proposed A2C architecture training on tilt-rotor problem. The learning graph shows a positive trend. It took 3 days to complete the training on Ryzen 9 5950x CPU with Nvidia GTX 3080 GPU.
![image](https://user-images.githubusercontent.com/55484402/146600203-2659ef11-e428-4d82-b571-cd79d89f7c14.png)

### Solved Normal Rotor using A2C
https://user-images.githubusercontent.com/55484402/146594520-69391b44-2cdc-4fb5-9a71-f12d904723aa.mp4

### Solved Tilt Rotor using A2C
https://user-images.githubusercontent.com/55484402/146599816-07d71855-dc6d-4821-a631-6e9e8d4a7e9a.mp4


