# Quantum machine learning for the portfolio

This note explains `intermediate_portfolio_qml.ipynb`. The intermediate challenge asks for a larger constrained portfolio and one chosen route. This route is a variational quantum model, not a deeper QAOA.

## The problem

There are \(n = 12\) assets. A portfolio is a bitstring \(x \in \{0,1\}^{12}\) with exactly \(K = 4\) ones. The objective to maximize is

\[
f(x) = \sum_i \mu_i x_i - q \sum_{i,j} \sigma_{ij} x_i x_j,
\qquad q = 2.
\]

The covariance is one-factor, \(\sigma_{ij} = 0.03\,\beta_i\beta_j + \delta_{ij}\omega_i^2\). A high \(\beta\) raises both the return and the cost of holding that asset together with other high-\(\beta\) assets. There are \(\binom{12}{4} = 495\) feasible portfolios, so brute force is exact and is used only as the test score.

## Why this is machine learning

QAOA minimizes a cost Hamiltonian. Here the circuit is a model with trainable angles \(\theta\). Measuring it defines a distribution \(P_\theta(x)\) over portfolios. Training moves that distribution toward high values of \(f\).

| Piece | Role |
|---|---|
| Model | One qubit per asset. A fixed feature rotation, then two layers of \(\mathrm{CZ}\) and trainable \(R_Y R_Z\). |
| Training data | All 495 feasible portfolios. Each one is labeled by its own \(f(x)\), not by a flag that says it is optimal. |
| Loss | \(L(\theta) = -\sum_{\|x\|_1=K} P_\theta(x)\, e^{25\, f(x)}\). Illegal bitstrings contribute nothing. |

\(f\) is known classically, so the model is not discovering a hidden label. What training changes is which legal portfolio carries the most probability.

## The circuit

Qubit \(i\) is asset \(i\).

**Feature map.** The marginal score \(s_i = \mu_i - q\sigma_{ii}\) is scaled into an angle \(\phi_i \in (0.2\pi,\, 0.8\pi)\) and applied as \(R_Y(\phi_i)\):

\[
R_Y(\phi)|0\rangle = \cos(\phi/2)\,|0\rangle + \sin(\phi/2)\,|1\rangle,
\qquad P(x_i=1) = \sin^2(\phi_i/2).
\]

A high-score asset starts more likely to be selected. At \(\theta = 0\) the trainable rotations are the identity, and \(\mathrm{CZ}\) does not change computational-basis probabilities, so the most likely weight-4 string is exactly the marginal greedy portfolio: assets 3, 6, 7, and 11, with \(f = -0.1688\).

**Two trainable layers.** Each layer is a line of \(\mathrm{CZ}\) on neighbouring assets, then \(R_Y(\theta)\) and \(R_Z(\theta)\) on every qubit. There are \(2 \times 12 \times 2 = 48\) parameters. The \(\mathrm{CZ}\) writes a phase when two neighbours are both selected. The following \(R_Y\) is the gate that can turn that phase into a different probability. \(R_Z\) only changes phase.

The budget is not a penalty term. Any measured string whose number of ones is not 4 is discarded. The reported portfolio is the legal string with the largest probability.

## Training

COBYLA minimizes \(L(\theta)\). It is local, so the run keeps the best of a zero start and five random starts (`maxiter` 120). The exponential in the loss matters: a shift such as \(f - f_{\min}\) makes a mediocre portfolio almost as valuable as the best one, and the optimizer then piles probability on the mediocre set. With \(\alpha = 25\), \(e^{\alpha f}\) drops as soon as \(f\) falls below the optimum.

On the statevector the trained mode is assets **0, 3, 4, and 6**, with

\[
f = f^\star = 0.0594, \qquad \text{approximation ratio } 1.0.
\]

About 46% of the probability sits on weight-4 strings, and 38% sits on this one portfolio. The mean of \(f/f^\star\) over all legal outcomes is 0.433. That average and the mode are different objects: the average still includes poor feasible portfolios that have a little amplitude.

## Classical comparison

| Rule | Assets | \(f\) | \(f/f^\star\) |
|---|---|---|---|
| Marginal greedy | 3, 6, 7, 11 | −0.1688 | −2.84 |
| Swap search | 0, 3, 4, 6 | 0.0594 | 1.00 |
| Trained circuit | 0, 3, 4, 6 | 0.0594 | 1.00 |

Marginal greedy ignores \(\sigma_{ij}\) for \(i \neq j\). Swap search starts there and exchanges one held asset for one excluded asset until \(f\) stops rising. It also finds the optimum. The quantum model ties that search and beats the ranking it was initialized to. On this instance, learning is the move from assets 3, 6, 7, 11 to assets 0, 3, 4, 6.

## The device

Training calls the circuit hundreds of times, so those calls use a noiseless statevector. One trained circuit was sampled on `ibm_basquecountry`, a 156-qubit Heron. The circuit uses 12 of those qubits. Job `dbm6o2e0fc9c73dhnev0` took 4096 shots.

Of those shots, 1697 (41.4%) have exactly four ones. The most frequent legal outcome, 1281 shots, is again assets 0, 3, 4, and 6, with ratio 1.0. Routing onto the heavy-hex chip and device noise did not move the mode off the optimum.

Bitstrings from the sampler are little-endian: the rightmost character is qubit 0.

## What this does not show

Brute force over 495 portfolios is exact. A 12-qubit statevector plus one device sample is not a quantum advantage. A swap search reaches the same portfolio. The result that the route has to justify is that a shallow variational circuit, trained on \(f\) and started from the marginal scores, reproduces that portfolio on the simulator and on the device.
