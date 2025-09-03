# Mathematical Formulation of Strategy Algorithms

## 1. Attribute Ordering

Given relative frequencies $f_a$ for each attribute $a \in A$, order attributes by decreasing frequency:

$$\text{attribute\_order} = \sort(A, \text{key} = f_a, \text{descending})$$

## 2. Constraint Sets

For each attribute $a$ with minimum count requirement $c_a^{\min}$, compute:

$$\alpha_a = \frac{c_a^{\min}}{N}$$

$$S_a = \{ \mathbf{t} \in \{0,1\}^K \mid t_{\text{index}(a)} = 1 \}$$

where $\mathbf{t}$ represents a type vector and $\text{index}(a)$ is the position of attribute $a$ in the ordered list.

## 3. Joint Probability Estimation with Correlations

### Gaussian Copula Approach

Given:
- Attribute order: $a_1, a_2, \dots, a_K$
- Marginal frequencies: $f_{a_i}$ for $i = 1, 2, \dots, K$
- Correlation matrix: $\rho_{ij}$ between attributes $a_i$ and $a_j$

**Step 1: Build correlation matrix**

$$\mathbf{R} = \begin{pmatrix}
1 & \rho_{12} & \cdots & \rho_{1K} \\
\rho_{21} & 1 & \cdots & \rho_{2K} \\
\vdots & \vdots & \ddots & \vdots \\
\rho_{K1} & \rho_{K2} & \cdots & 1
\end{pmatrix}$$

**Step 2: Ensure positive semi-definiteness**

If any eigenvalue $\lambda_i < 0$, adjust:
$$\mathbf{R} \leftarrow \mathbf{R} + (-\min(\lambda_i) + \epsilon)\mathbf{I}$$

**Step 3: Generate correlated samples**

$$\mathbf{z}^{(m)} \sim \mathcal{N}(\mathbf{0}, \mathbf{R}) \quad \text{for } m = 1, 2, \dots, M$$

**Step 4: Convert to binary using inverse CDF**

For each attribute $i$:
$$\theta_i = \sqrt{2} \cdot \text{erfinv}(2f_{a_i} - 1)$$

**Step 5: Generate type indicators**

$$t_i^{(m)} = \mathbb{I}[z_i^{(m)} > \theta_i]$$

**Step 6: Estimate probabilities**

$$p_{\mathbf{t}} = \frac{1}{M} \sum_{m=1}^M \mathbb{I}[\mathbf{z}^{(m)} \mapsto \mathbf{t}]$$

where $\mathbf{t} = (t_1, t_2, \dots, t_K)$ is the type vector.

## 4. Multiplicative Weights Algorithm

### Problem Formulation

Maximize: $\sum_{\mathbf{t}} p_{\mathbf{t}} a_{\mathbf{t}}$

Subject to:
$$\sum_{\mathbf{t} \in S_a} p_{\mathbf{t}} a_{\mathbf{t}} \geq \alpha_a \quad \forall a \in A$$
$$0 \leq a_{\mathbf{t}} \leq 1 \quad \forall \mathbf{t}$$

### Algorithm

**Initialize:** $a_{\mathbf{t}} = 1$ for all $\mathbf{t}$

**For iteration $k = 1$ to $T$:**
- Compute slacks: $s_a = \sum_{\mathbf{t}} (I_a(\mathbf{t}) - \alpha_a) p_{\mathbf{t}} a_{\mathbf{t}}^k$
- If $s_a \geq 0$ for all $a$: break
- Update: $a_{\mathbf{t}}^{k+1} = a_{\mathbf{t}}^k \cdot \prod_{a: s_a < 0, \mathbf{t} \notin S_a} \exp(-\eta p_{\mathbf{t}})$
- Clip: $a_{\mathbf{t}}^{k+1} = \max(0, \min(1, a_{\mathbf{t}}^{k+1}))$

where $I_a(\mathbf{t}) = 1$ if $\mathbf{t} \in S_a$, 0 otherwise.

## 5. Online Admission Policy

### State Variables

- $n$: total accepted applicants
- $c_a$: count of accepted applicants with attribute $a$
- $r_a$: remaining required for constraint $a$: $r_a = \max(0, \lceil \alpha_a N \rceil - c_a)$

### Decision Logic

**Input:** Applicant attributes $\mathbf{x} = (x_1, x_2, \dots, x_K)$

**Remaining capacity:** $R = N - n$

#### Phase 1: Fill remaining capacity (if all constraints satisfied)

If $\sum_a r_a = 0$ and $R > 0$: Accept

#### Phase 2: Forced mode (endgame)

$$\text{forced\_mode} = (\sum_a r_a > R) \lor (\exists a: r_a = R)$$

If forced mode:
- If $\exists a: r_a > 0 \land x_a = 1$: Accept
- Else: Reject

#### Phase 3: Deficit-aware scoring (normal operation)

**Buffer-aware alpha:**
$$\alpha_a' = \begin{cases}
\alpha_a + \beta / \sqrt{N} & \text{if } n < 0.8N \\
\alpha_a & \text{otherwise}
\end{cases}$$

**Deficit for each constraint:**
$$d_a = \max(0, \alpha_a' - c_a / \max(1, n))$$

**Score for applicant:**
$$s = \sum_{a: x_a = 1} d_a$$

**Acceptance probability:**
$$\pi = \min(1, p_{\mathbf{t}} \cdot (1 + \lambda \cdot s))$$

where $\mathbf{t}$ is the type corresponding to $\mathbf{x}$.

**Decision:** Accept with probability $\pi$

## 6. Parallel Processing Enhancements

### Chunk-based Processing

**Sample generation:** Generate $M$ samples at once
$$\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, \mathbf{R})^{\otimes M}$$

**Parallel processing:** Split $\mathbf{Z}$ into $W$ chunks
$$\mathbf{Z}_w \subset \mathbf{Z} \quad \text{for } w = 1, 2, \dots, W$$

**Worker computation:** Each worker processes chunk $w$
$$\text{counts}_w = \sum_{\mathbf{z} \in \mathbf{Z}_w} \mathbb{I}[\mathbf{z} \mapsto \mathbf{t}]$$

**Aggregation:** Combine results across workers
$$\text{total\_counts}_{\mathbf{t}} = \sum_w \text{counts}_{w,\mathbf{t}}$$

### Vectorized Operations

**Matrix form conversion:**
$$\mathbf{T} = \mathbb{I}[\mathbf{Z} > \boldsymbol{\theta}^T]$$

**Efficient counting:**
$$\text{counts} = \text{unique}(\mathbf{T}, \text{axis}=0, \text{return\_counts}=\text{True})$$

## 7. Key Parameters

- $N$: Total capacity
- $K$: Number of attributes
- $M$: Number of samples for probability estimation
- $\eta$: Learning rate for multiplicative weights
- $\lambda$: Nudge parameter for online policy
- $\beta$: Safety buffer parameter
- $\epsilon$: Small constant for numerical stability

## 8. Performance Characteristics

**Time Complexity:**
- Joint probability estimation: $O(M \cdot K + K^3)$ (Gaussian sampling + correlation matrix)
- Multiplicative weights: $O(T \cdot |T| \cdot |A|)$ where $|T| = 2^K$
- Online decision: $O(|A|)$ per applicant

**Space Complexity:**
- Type probabilities: $O(2^K)$
- Correlation matrix: $O(K^2)$
- Policy state: $O(|A|)$

This formulation provides a complete mathematical foundation for the strategy algorithms, enabling rigorous analysis and potential optimization of the admission control system.
