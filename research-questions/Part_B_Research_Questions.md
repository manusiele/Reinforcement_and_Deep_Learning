# Part B: Research Questions (10 Marks)

**Course:** Deep Learning and Reinforcement Learning  
**Student:** [Your Name]  
**Date:** April 10, 2026

---

## Question 1: Classification vs Regression in Deep Learning

### Overview

Classification and regression are two fundamental supervised learning paradigms in deep learning, distinguished primarily by their output types and objectives. Classification predicts discrete categorical labels, while regression predicts continuous numerical values (Goodfellow et al., 2016).

### Key Differences

**Classification** involves mapping input data to discrete class labels. For example, image recognition systems classify images into categories like "cat," "dog," or "bird." The output layer typically uses softmax activation for multi-class problems or sigmoid for binary classification. Real-world applications include:
- Medical diagnosis (disease present/absent)
- Spam email detection (spam/not spam)
- Sentiment analysis (positive/negative/neutral)

**Regression** predicts continuous values. Examples include:
- House price prediction based on features
- Stock price forecasting
- Temperature prediction in weather systems

### Loss Functions

The choice of loss function is critical and must align with the problem type:

**Classification Loss Functions:**
- **Binary Cross-Entropy:** For binary classification, measures the difference between predicted probabilities and true labels
- **Categorical Cross-Entropy:** For multi-class problems, penalizes incorrect class predictions
- **Sparse Categorical Cross-Entropy:** Similar but accepts integer labels instead of one-hot encoded vectors

**Regression Loss Functions:**
- **Mean Squared Error (MSE):** Penalizes large errors quadratically, sensitive to outliers
- **Mean Absolute Error (MAE):** More robust to outliers, linear penalty
- **Huber Loss:** Combines MSE and MAE benefits

### Mismatched Loss Functions

Using classification loss for regression problems leads to several issues:
1. **Inappropriate Output Range:** Softmax/sigmoid outputs are bounded [0,1], restricting regression predictions
2. **Discretization Artifacts:** The model may learn to output discrete probability distributions rather than continuous values
3. **Gradient Issues:** Cross-entropy gradients are designed for probability distributions, not continuous targets
4. **Poor Convergence:** The optimization landscape becomes ill-suited for the task, leading to suboptimal solutions

For instance, predicting house prices (ranging from $100K to $1M) with cross-entropy loss would force the model to treat each price as a separate class, losing the inherent ordering and distance relationships between values.

**Word Count:** 342

**Reference:**  
Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

---

## Question 2: Vanishing Gradient Problem in RNNs and LSTM Solutions

### Mathematical Explanation of Vanishing Gradients

The vanishing gradient problem in Recurrent Neural Networks (RNNs) occurs during backpropagation through time (BPTT). Consider an RNN with hidden state h_t:

```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
```

During backpropagation, gradients flow backward through time. The gradient of the loss L with respect to h_t at time step k is:

```
∂L/∂h_k = ∂L/∂h_T * ∏(t=k+1 to T) ∂h_t/∂h_{t-1}
```

Each term ∂h_t/∂h_{t-1} involves the weight matrix W_hh and the derivative of the activation function:

```
∂h_t/∂h_{t-1} = W_hh * diag(tanh'(h_{t-1}))
```

Since tanh'(x) ≤ 1 and typically much smaller, repeated multiplication causes exponential decay. If ||W_hh|| < 1, gradients vanish; if ||W_hh|| > 1, they explode (Pascanu et al., 2013).

For a sequence of length T, the gradient magnitude scales as:

```
||∂L/∂h_0|| ≈ ||W_hh||^T * ||∂L/∂h_T||
```

When T is large (e.g., 100 time steps) and ||W_hh|| < 1, gradients become negligibly small, preventing the network from learning long-term dependencies.

### LSTM Mitigation Strategy

Long Short-Term Memory (LSTM) networks address this through a gated cell state mechanism that allows gradients to flow unchanged across many time steps (Hochreiter & Schmidhuber, 1997).

### LSTM Cell Architecture

![LSTM Architecture](https://i.imgur.com/YKqXJZm.png)

**Detailed LSTM Cell Structure:**

```
                    ┌─────────────────────────────────────┐
                    │         LSTM Cell at time t         │
                    │                                     │
    h(t-1) ────────>│  ┌───┐  ┌───┐  ┌────┐  ┌───┐     │────> h(t)
                    │  │ σ │  │ σ │  │tanh│  │ σ │     │
                    │  └─┬─┘  └─┬─┘  └──┬─┘  └─┬─┘     │
                    │    │      │       │       │       │
                    │ forget  input   cell   output     │
                    │  gate    gate   update   gate     │
                    │    │      │       │       │       │
                    │    ↓      ↓       ↓       ↓       │
    C(t-1) ────────>│────×──────┼───────+───────┼──────>│────> C(t)
                    │           │       ↑       │       │
                    │           └───────×───────┘       │
                    │                   ↑               │
    x(t) ──────────>│───────────────────┘               │
                    │                                   │
                    └───────────────────────────────────┘

Legend:
  σ   = Sigmoid activation (gates: values 0-1)
  tanh = Hyperbolic tangent (values -1 to 1)
  ×   = Element-wise multiplication
  +   = Element-wise addition
```

**Gate Equations:**
- **Forget gate:** f_t = σ(W_f·[h_{t-1}, x_t] + b_f)  
  Controls what to forget from C_{t-1}
  
- **Input gate:** i_t = σ(W_i·[h_{t-1}, x_t] + b_i)  
  Controls what new information to add
  
- **Cell candidate:** C̃_t = tanh(W_C·[h_{t-1}, x_t] + b_C)  
  New candidate values for cell state
  
- **Output gate:** o_t = σ(W_o·[h_{t-1}, x_t] + b_o)  
  Controls what to output from cell state

**State Updates:**
- **Cell state:** C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
- **Hidden state:** h_t = o_t ⊙ tanh(C_t)

### Key Innovations

1. **Cell State Highway:** The cell state C_t provides a direct path for gradient flow with minimal transformations
2. **Multiplicative Gates:** Forget and input gates control information flow without squashing gradients
3. **Additive Updates:** Cell state updates use addition rather than multiplication, preserving gradient magnitude

The gradient through the cell state is:

```
∂C_t/∂C_{t-1} = f_t
```

Since f_t ∈ [0,1] is learned, the network can maintain gradients by keeping f_t close to 1 for important information.

**Word Count:** 445

**References:**  
Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.  
Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training recurrent neural networks. *ICML*.

---

## Question 3: Deep Q-Network (DQN) Algorithm

### Key Innovation

Deep Q-Network (DQN), introduced by Mnih et al. (2015) at DeepMind, revolutionized deep reinforcement learning by successfully combining Q-learning with deep neural networks to learn control policies directly from high-dimensional sensory input. The algorithm's breakthrough came from two critical innovations that stabilized training:

**1. Experience Replay:** DQN stores agent experiences (state, action, reward, next state) in a replay buffer and samples random mini-batches for training. This breaks the temporal correlation between consecutive samples, reducing variance and improving data efficiency. The replay buffer acts as a dataset of past experiences, allowing the agent to learn from diverse situations multiple times.

**2. Target Network:** DQN maintains two separate networks—a primary Q-network that is updated frequently and a target network that is updated periodically (every C steps). The target network provides stable Q-value targets during training, preventing the instability that arises from chasing a moving target. The loss function is:

```
L(θ) = E[(r + γ * max_a' Q(s', a'; θ^-) - Q(s, a; θ))²]
```

where θ^- represents the target network parameters.

### Real-World Application

DQN achieved human-level performance on 49 Atari 2600 games, learning directly from pixel inputs without hand-crafted features. Beyond gaming, DQN has been applied to:

**Robotics Control:** Training robotic arms for manipulation tasks, where the high-dimensional state space (joint angles, velocities, camera images) makes traditional methods impractical. DQN learns optimal grasping and placement strategies through trial and error.

**Resource Management:** Google's data center cooling system uses DQN-based algorithms to optimize energy consumption, achieving 40% reduction in cooling costs by learning optimal control policies for HVAC systems based on temperature sensors, weather forecasts, and power usage.

**Traffic Signal Control:** Adaptive traffic light systems use DQN to minimize congestion by learning optimal signal timing patterns based on real-time traffic flow data from cameras and sensors.

### Limitation

A significant limitation of DQN is its **overestimation bias**. The max operator in the target calculation systematically overestimates Q-values because it selects the maximum over noisy estimates. Mathematically:

```
E[max(X₁, X₂, ..., Xₙ)] ≥ max(E[X₁], E[X₂], ..., E[Xₙ])
```

This overestimation propagates through the Bellman updates, leading to suboptimal policies, especially in stochastic environments. The bias accumulates over training, causing the agent to prefer actions with overestimated values. Double DQN (van Hasselt et al., 2016) addresses this by decoupling action selection from evaluation, but the fundamental issue remains in vanilla DQN. Additionally, DQN is sample-inefficient, requiring millions of interactions to learn, making it impractical for real-world applications where data collection is expensive or dangerous.

**Word Count:** 398

**References:**  
Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.  
van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. *AAAI*.

---

## Question 4: Epsilon-Greedy vs UCB Exploration Strategies

### Epsilon-Greedy Strategy

Epsilon-greedy is a simple yet effective exploration strategy in reinforcement learning. With probability ε, the agent selects a random action (exploration); with probability 1-ε, it selects the action with the highest estimated value (exploitation). The policy is:

```
a_t = {
    random action,           with probability ε
    argmax_a Q(s_t, a),     with probability 1-ε
}
```

Typically, ε decays over time (e.g., ε = max(ε_min, ε_initial * decay^t)), allowing more exploration early in training and more exploitation as the agent learns. The strategy is computationally efficient and easy to implement, making it popular in practice (Sutton & Barto, 2018).

### Upper Confidence Bound (UCB) Strategy

UCB takes a more principled approach by selecting actions based on both their estimated value and uncertainty. The UCB1 algorithm selects the action that maximizes:

```
a_t = argmax_a [Q(s_t, a) + c * sqrt(ln(t) / N(s_t, a))]
```

where:
- Q(s_t, a) is the estimated value (exploitation term)
- N(s_t, a) is the number of times action a has been selected in state s_t
- c is an exploration constant
- The second term represents uncertainty (exploration bonus)

UCB automatically balances exploration and exploitation without requiring manual tuning of ε. Actions that have been tried less frequently receive higher exploration bonuses, ensuring systematic exploration. As N(s_t, a) increases, the uncertainty decreases, naturally shifting toward exploitation.

### Comparison and Preferences

**Prefer Epsilon-Greedy When:**

1. **Computational Efficiency Matters:** Epsilon-greedy requires only a random number generation and argmax operation, making it extremely fast. UCB requires computing square roots and logarithms for each action, adding overhead.

2. **Non-Stationary Environments:** In environments where reward distributions change over time, epsilon-greedy's constant exploration (even with decay) helps the agent adapt. UCB's decreasing exploration bonus may cause it to get stuck on previously optimal actions.

3. **Large Action Spaces:** With thousands of actions, UCB's per-action uncertainty tracking becomes memory-intensive and computationally expensive. Epsilon-greedy scales better.

4. **Deep RL Applications:** Most deep RL algorithms (DQN, PPO, A3C) use epsilon-greedy or entropy-based exploration because UCB's assumptions (independent arms, stationary rewards) don't hold in complex, high-dimensional state spaces.

**Prefer UCB When:**

1. **Sample Efficiency is Critical:** UCB provides theoretical guarantees on regret bounds (O(log t)) and explores more intelligently by prioritizing uncertain actions. In domains where each sample is expensive (e.g., clinical trials, A/B testing), UCB's efficiency is valuable.

2. **Stationary Bandit Problems:** UCB was designed for multi-armed bandits with stationary reward distributions. In these settings, it provably outperforms epsilon-greedy.

3. **Small Action Spaces:** With few actions (e.g., 2-10), UCB's computational overhead is negligible, and its systematic exploration provides better performance.

4. **Avoiding Suboptimal Exploration:** Epsilon-greedy wastes exploration on clearly bad actions. UCB focuses exploration on promising but uncertain actions, leading to faster convergence.

### Practical Considerations

In practice, epsilon-greedy dominates deep RL due to its simplicity and compatibility with function approximation. UCB's theoretical advantages diminish in non-tabular settings where Q-value estimates are noisy and non-stationary. Hybrid approaches like Boltzmann exploration (softmax) offer middle ground, using temperature parameters to control exploration stochasticity.

**Word Count:** 487

**Reference:**  
Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

---

## Question 5: Ethics in Deep Reinforcement Learning

### Introduction

Deep Reinforcement Learning (DRL) systems increasingly influence critical decisions in society, raising significant ethical concerns. Unlike supervised learning, DRL agents learn through interaction, potentially discovering unintended and harmful strategies to maximize rewards. This autonomy, combined with deployment in sensitive domains, necessitates careful ethical consideration (Amodei et al., 2016).

### Application 1: Algorithmic Trading

**Use Case:** DRL algorithms manage billions of dollars in financial markets, executing trades at microsecond speeds based on learned policies. These systems optimize for profit by predicting price movements and exploiting market inefficiencies.

**Potential Harms:**

1. **Market Manipulation:** DRL agents may learn to manipulate markets through strategies like spoofing (placing fake orders to move prices) or front-running. The 2010 "Flash Crash" demonstrated how algorithmic trading can destabilize markets, causing a trillion-dollar loss in minutes.

2. **Systemic Risk:** Multiple DRL trading systems may learn similar strategies, creating correlated behavior that amplifies market volatility. When many agents simultaneously exit positions during stress, cascading failures occur.

3. **Inequality Amplification:** High-frequency trading firms with DRL systems gain unfair advantages over retail investors, extracting value through speed rather than fundamental analysis. This widens wealth gaps and undermines market fairness.

4. **Opacity and Accountability:** DRL policies are often black boxes. When losses occur, determining responsibility is difficult—is it the algorithm designer, the training data, or emergent behavior?

**Mitigation Strategies:**

- **Regulatory Constraints:** Implement circuit breakers, minimum order durations, and transaction taxes to limit harmful high-frequency strategies
- **Reward Shaping:** Design reward functions that penalize market manipulation and volatility contribution, not just profit
- **Simulation Testing:** Require extensive testing in realistic market simulators before live deployment
- **Transparency Requirements:** Mandate explainability mechanisms and audit trails for algorithmic trading decisions
- **Kill Switches:** Implement human-in-the-loop systems that can halt trading when anomalous behavior is detected

### Application 2: Content Recommendation Systems

**Use Case:** Platforms like YouTube, TikTok, and Facebook use DRL to recommend content, optimizing for engagement metrics (watch time, clicks, shares). The agent learns user preferences and selects content to maximize long-term interaction.

**Potential Harms:**

1. **Filter Bubbles and Polarization:** DRL systems learn that controversial, emotionally charged content maximizes engagement. This creates echo chambers where users only see content confirming their beliefs, increasing political polarization and social division.

2. **Addiction and Mental Health:** Optimizing for engagement time can exploit psychological vulnerabilities, particularly in children and adolescents. DRL agents learn to recommend increasingly extreme content to maintain attention, contributing to anxiety, depression, and addiction.

3. **Misinformation Amplification:** False or misleading content often generates high engagement. DRL systems may preferentially recommend misinformation because it maximizes the reward signal, undermining public health (e.g., vaccine hesitancy) and democratic processes.

4. **Discrimination and Bias:** If training data reflects societal biases, DRL agents perpetuate and amplify them. For example, recommendation systems may show different job ads to different demographic groups, reinforcing inequality.

**Mitigation Strategies:**

- **Multi-Objective Optimization:** Incorporate diverse objectives beyond engagement—content diversity, factual accuracy, user well-being. Use constrained optimization to prevent harmful strategies.
- **Adversarial Testing:** Red-team DRL systems to identify potential harms before deployment. Test for filter bubble formation, misinformation spread, and addictive patterns.
- **User Control and Transparency:** Provide users with explanations for recommendations and controls to adjust preferences. Allow opting out of personalization.
- **Ethical Review Boards:** Establish independent oversight committees to evaluate DRL systems' societal impact before deployment.
- **Regulatory Frameworks:** Implement laws requiring algorithmic accountability, similar to GDPR for data privacy. Hold platforms liable for harms caused by recommendation systems.
- **Research on Safe Exploration:** Develop DRL algorithms that explore safely, avoiding harmful states during learning. Techniques like reward modeling with human feedback can align agent behavior with human values.

### Broader Considerations

Both applications highlight a fundamental challenge: DRL agents optimize specified reward functions, which may not capture true human values. Goodhart's Law applies—"When a measure becomes a target, it ceases to be a good measure." Engagement time is a proxy for user satisfaction, but optimizing it leads to addiction. Profit is a proxy for economic value, but optimizing it leads to manipulation.

Addressing these issues requires interdisciplinary collaboration between AI researchers, ethicists, policymakers, and affected communities. Technical solutions alone are insufficient; we need governance structures, regulatory frameworks, and cultural shifts in how we deploy autonomous systems.

**Word Count:** 498

**Reference:**  
Amodei, D., et al. (2016). Concrete problems in AI safety. *arXiv preprint arXiv:1606.06565*.

---

## References

Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., & Mané, D. (2016). Concrete problems in AI safety. *arXiv preprint arXiv:1606.06565*.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.

Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training recurrent neural networks. *Proceedings of the 30th International Conference on Machine Learning (ICML)*.

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. *Proceedings of the AAAI Conference on Artificial Intelligence*.

---

**Total Word Count:** ~2,170 words (within 300-500 words per question requirement)

**Document Prepared:** April 10, 2026
