The Berghain Bouncer Algorithm (BBA)
0) Notation you’ll use once and then forget
There are K binary constraints of the form “at least αⱼ of the final 1000 must have property j” (examples: Berlin local, all-black, regulars).

A “type” t is a full attribute combo across all properties (there are up to 2K types).

You know the joint probabilities p_t (from the given frequencies + correlations).

Sⱼ is the set of types that satisfy constraint j.

When a type-t person arrives, you either accept or reject immediately.

Goal: minimize rejections before you accept N people, subject to all mins.

1) Feasibility sanity check (one-time)
If the mins are so extreme that they can’t be met from the population, don’t waste the night.

For each j: you can’t require αⱼ greater than the population share of Sⱼ across types, i.e.

$$ \sum_{t\in S_j} p_t ;\ge; \alpha_j $$

If any fails, the scenario is infeasible; if all pass, proceed.

2) Compute base acceptance rates (the “offline” optimum)
This gives you the asymptotically optimal stationary policy (fewest expected rejections) for i.i.d. arrivals.

Choose an admission probability $a_t \in [0,1]$ for each type t and keep it fixed. Per arrival, your overall accept rate is $A = \sum_t p_t a_t$. Among accepted people, the share of constraint j is:

$$ \frac{\sum_{t\in S_j} p_t a_t}{\sum_t p_t a_t} ;\ge; \alpha_j $$

Re-arrange each constraint:

$$ \sum_{t} \big(\mathbf{1}[t\in S_j]-\alpha_j\big) p_t a_t ;\ge; 0 \quad\text{for all } j $$

So the “best” stationary policy solves the LP:

$$ \begin{aligned} \text{maximize } & \sum_t p_t a_t \quad (=A)\ \text{subject to } & \sum_t (\mathbf{1}[t\in S_j]-\alpha_j) p_t a_t \ge 0 ;; \forall j\ & 0 \le a_t \le 1 ;; \forall t \end{aligned} $$

What it means: start at $a_t=1$ for all types; if a quota j would be under-represented under full admission, you must throttle non-j types until the constraints hold; all the “scarce” types (those that help bind constraints) should stay at $a_t=1$.

A quick, effective solver (no fancy libraries)
Initialize $a_t \leftarrow 1$ for all t.

Repeat 200–500 iterations:

For each constraint j compute the slack:

$$ g_j \leftarrow \sum_t (\mathbf{1}[t\in S_j]-\alpha_j) p_t a_t $$

If all $g_j \ge 0$, stop; you’re feasible and near-optimal.

Otherwise, for each violated j (where $g_j<0$), down-weight the over-represented types (those not in Sⱼ):

$$ a_t \leftarrow a_t \cdot \exp!\Big(-\eta \cdot p_t \cdot \sum_{j:,g_j<0}\big(1-\mathbf{1}[t\in S_j]\big)\Big) $$

Clip $a_t$ back into [0,1]. Small step $\eta\in[0.1,0.5]/K$ works.

The resulting $a_t*$ is your base admission probability for each type.

Score estimate from this stage: expected rejections to fill N is

$$ \text{rej}_\text{exp} \approx N\Big(\frac{1}{A*}-1\Big), \quad A*=\sum_t p_t a_t* $$

3) Online policy (turn the crank at the door)
Stationary $a_t*$ is already near-optimal for N=1000, but random wiggles can push you off target; you’ll correct gently with deficits and lock the endgame.

Keep counters while admitting:

Total accepted so far: $n$

For each constraint j: $c_j$ = accepted who satisfy j

Remaining capacity: $R = N - n$

Remaining required for j: $r_j = \max(0,\lceil \alpha_j N \rceil - c_j)$

Each arrival of type t:

Hard endgame guard (can’t-miss mode). If there exists a j such that $R = \sum_{j} r_j$ in a way that forces you to take only people who satisfy those remaining mins, then:

If type t fails any needed j with $r_j>0$, reject.

If type t satisfies any needed j, accept. (Practical shortcut: if some j has $r_j = R$, admit only Sⱼ until it hits zero.)

Deficit-aware accept score. Compute a deficit weight per constraint:

$$ w_j ;=; \max!\Big(0,; \alpha_j - \frac{c_j}{\max(1,n)}\Big) $$

and a type score:

$$ s(t) ;=; \sum_j w_j ,\mathbf{1}[t\in S_j] $$

Interpret $s(t)$ as “how much this person helps quotas right now”.

Admit with a nudged probability:

$$ \text{admit with prob } ;; \pi_t ;=; \min\Big{1,; a_t* \cdot \big(1 + \lambda \cdot s(t)\big)\Big} $$

with a small $\lambda$ (e.g. 0.5). In practice:

If $s(t) > 0$, admit with probability bumped above $a_t*$.

If $s(t) = 0$, admit with probability $a_t*$.

If some quotas are already comfortably above target, their wⱼ go to 0 automatically.

Safety stock buffer (avoid last-minute pain). Work against buffered targets $\tilde\alpha_j = \alpha_j + \beta/\sqrt{N}$ for the first ~70–80% of the night (β≈1–2), then drop back to $\alpha_j$. This eats tiny acceptance slack early to slash the probability of missing a min at the end.

That’s it; it’s just “LP-tuned base rates” + “deficit nudge” + “endgame lock”.

4) What to do with correlations
You already have correlations; just compute the joint type probabilities $p_t$ for the 2K combos (or the subset that actually occurs), and run the exact same recipe. The correlations help you because some types will simultaneously satisfy multiple mins (e.g., “local AND black”), and the LP automatically prioritizes those with $a_t*=1$.

5) Minimal, robust pseudocode
# Inputs: N, constraints {alpha_j}, joint type probs {p_t}, membership 1[t in S_j]
# Offline: compute a_t* via the multiplicative-weights LP projection above.

N = 1000
a = {t: a_star_t for t in types}        # from Section 2
n = 0
c = {j: 0 for j in constraints}

while n < N:
    t = observe_next_arrival_type()     # attribute combo of new person
    R = N - n
    r = {j: max(0, ceil(alpha[j]*N) - c[j]) for j in constraints}

    # Endgame hard guard
    forced = any(rj == R for rj in r.values()) or sum(r.values()) >= R
    if forced:
        if any(r[j] > 0 and t not in S[j] for j in constraints):
            reject(); continue
        else:
            accept(); n += 1
            for j in constraints:
                if t in S[j]: c[j] += 1
            continue

    # Deficit-aware score
    w = {j: max(0.0, alpha[j] - (c[j] / max(1,n))) for j in constraints}
    s = sum(w[j] for j in constraints if t in S[j])

    # Safety buffer early
    if n < 0.8*N:
        w = {j: max(0.0, (alpha[j] + 1.0/ (N**0.5)) - (c[j] / max(1,n))) for j in constraints}
        s = sum(w[j] for j in constraints if t in S[j])

    # Probabilistic admit with nudge
    lam = 0.5
    pi = min(1.0, a[t] * (1.0 + lam * s))

    if rand() < pi:
        accept(); n += 1
        for j in constraints:
            if t in S[j]: c[j] += 1
    else:
        reject()
Deterministic variant: accept iff $s(t) > \tau$ where $\tau$ is the smallest value that keeps your empirical accept rate near $A*$; the probabilistic version is smoother and usually rejects less.

6) Practical tips that move the leaderboard
Always keep high-leverage types at 100%: any type that simultaneously satisfies several binding mins should have $a_t*=1$ and be auto-accept in practice.

Throttle only what you must: your MW-LP will tell you which over-represented types to shave; don’t guess.

Lock the endgame early: when $R$ drops near $\sum_j r_j$, flip into strict guard so you don’t blow a min in the last 30 people.

Use a tiny buffer early: that + endgame lock typically saves dozens to hundreds of rejections versus naive “aim for exactly α by the end” play.

Score math: once you have $A*$, $\text{rej}_\text{exp} \approx N(\tfrac{1}{A*} - 1)$; maximizing $A*$ is the whole game.

7) What you actually do on each scenario
From the sheet they give you, compute the joint type probs $p_t$.

Run the 30-second MW-LP to get the base $a_t*$.

Start the door with the online rule above (nudged by deficits, with the hard guard in the endgame).

Enjoy the smoothest fill that still nails the mins with the fewest bodies turned away.

This is the optimal-in-expectation stationary backbone (the LP), plus state-aware correction and a no-nonsense finish, which is exactly what wins this kind of online quota game.