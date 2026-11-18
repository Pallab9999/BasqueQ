# Constrained portfolio selection

Work from the BasQ Qiskit Fall Fest 2026, Track 2: optimization under a budget constraint. The task was one problem at three sizes. I understood the first two. I did not understand the third well enough to solve it.

## The problem

Choose a subset of assets that maximizes expected return and penalizes risk, while holding exactly \(K\) assets:

\[
f(x) = \sum_i \mu_i x_i - q \sum_{i,j} \sigma_{ij} x_i x_j, \qquad \sum_i x_i = K.
\]

\(x_i\) is 1 when asset \(i\) is held. The covariance is a one-factor model, \(\sigma_{ij} = c\,\beta_i\beta_j + \delta_{ij}\omega_i^2\). A high \(\beta\) raises both the return and the cost of holding that asset together with other high-\(\beta\) assets.

The three levels were open-ended. Small instances can be scored by brute force, so a quantum result can be checked.

| Level | What it asked for |
|---|---|
| Beginner | A small portfolio, encoded as a cost Hamiltonian with a penalty for the budget, solved with QAOA. Study depth \(p\) and the penalty strength. Check against brute force. |
| Intermediate | A larger portfolio, and one chosen route: either a harder QAOA, or a quantum machine-learning model. Beat a classical baseline and say what the learning changed. |
| Advanced | The same kind of problem at a size where shallow circuits stop working. Trainability, heavier constraints, and hardware-aware compilation. The prompt pointed at utility scale, on the order of 100 qubits. |

## What I did

### Beginner — QAOA, and this part is solved

Notebook: [`track_2_optimization/beginner_portfolio_maxcut_qaoa.ipynb`](track_2_optimization/beginner_portfolio_maxcut_qaoa.ipynb)

Instance: 6 assets, choose 3, risk weight \(q = 2\). There are 20 legal portfolios and 64 bitstrings, so brute force is exact. The optimum is assets 1, 3, and 4, with \(f^\star = 0.1364\).

The encoding is the usual chain, written out rather than hidden in a library call:

1. Penalized cost \(C(x) = -f(x) + \lambda(\sum x_i - K)^2\).
2. That polynomial is a QUBO.
3. Substitute \(x_i = (1 - Z_i)/2\) to get an Ising Hamiltonian.
4. Linear fields become edges to one extra reference qubit, fixed in \(\lvert 0\rangle\). The circuit is then a signed weighted Max-Cut. Some weights are negative, so it is not ordinary non-negative Max-Cut.

QAOA puts Hadamards on the asset qubits, an `RZZ` for every edge including the reference, and an `RX` mixer on the assets only. The reference is not mixed. Angles are trained with COBYLA on a statevector. The best evaluated energy is kept, because COBYLA's last proposal is not always its best.

At \(\lambda = 0.2\), about twenty times the smallest penalty that makes the optimum legal, the best feasible sample is the brute-force portfolio at depth \(p = 1, 2, 3\). The approximation ratio of that sample is 1. Raising \(p\) improves the expected energy and the fraction of shots that are legal. Below the minimum penalty, the ground state of \(C\) is illegal, and QAOA faithfully prefers that illegal state.

This checked result is noiseless. The angles were chosen on the simulator.

### Intermediate — quantum machine learning, and this part is solved

Notebook: [`track_2_optimization/intermediate_portfolio_qml.ipynb`](track_2_optimization/intermediate_portfolio_qml.ipynb)

Write-up: [`track_2_optimization/qml_explanation.md`](track_2_optimization/qml_explanation.md)

Slides: [`track_2_optimization/intermediate_qml_slides.pptx`](track_2_optimization/intermediate_qml_slides.pptx)

Instance: 12 assets, choose 4, \(q = 2\). There are 495 legal portfolios. Brute force still gives the test score: assets 0, 3, 4, and 6, with \(f^\star = 0.0594\).

I took the quantum-ML fork, not a deeper QAOA. One qubit is one asset. A feature \(R_Y\) is set from the marginal score \(\mu_i - q\sigma_{ii}\). Two layers then apply a line of `CZ` gates and a trainable \(R_Y R_Z\) on every qubit: 48 angles. Measuring the circuit defines a distribution over portfolios. Training minimizes

\[
L(\theta) = -\sum_{\lVert x\rVert_1 = 4} P_\theta(x)\, e^{25 f(x)}
\]

on a statevector, with COBYLA, from a zero start and five random starts. Strings that do not hold exactly four assets contribute nothing. There is no penalty term and no reference qubit. At \(\theta = 0\) the most likely legal string is the marginal greedy portfolio, assets 3, 6, 7, and 11, with \(f = -0.1688\).

After training, the mode is the brute-force optimum. About 46% of the probability is on weight-4 strings, and 38% is on that one portfolio.

| Rule | Assets | \(f / f^\star\) |
|---|---|---|
| Marginal greedy, before training | 3, 6, 7, 11 | −2.84 |
| Swap search | 0, 3, 4, 6 | 1.00 |
| Trained circuit | 0, 3, 4, 6 | 1.00 |
| Device, most frequent legal string | 0, 3, 4, 6 | 1.00 |

The device run is one frozen circuit, not the training loop. Job `dbm6o2e0fc9c73dhnev0` on `ibm_basquecountry` took 4096 shots of the 12-qubit circuit. Of those, 1697 have exactly four ones, and 1281 are the optimum. Routing and noise did not move the mode.

The classical limit is stated in the notebook. A swap search that starts from the greedy set and exchanges one asset at a time also finds the optimum. The quantum model beats the ranking it was initialized with. It ties the swap search. That is not a quantum advantage.

## Where it failed

The advanced level is not solved. I did not understand it.

The ask was to take this portfolio into the regime where the encoding and the training break: more assets, heavier constraints, a gradient that can actually be measured, and a circuit that survives the heavy-hex device. I could not turn that into a solver.

[`track_2_optimization/advanced_barren_plateaus.ipynb`](track_2_optimization/advanced_barren_plateaus.ipynb) is only a diagnosis on a simulator. It trains nothing about the portfolio. A random hardware-efficient circuit is differentiated with the parameter-shift rule, and the variance of one partial derivative is plotted for three known causes of a barren plateau:

- A global cost \(Z^{\otimes n}\) already decays at depth 2. From 4 to 12 qubits the variance falls by about \(2^{-n}\).
- A local cost \(\langle Z_0\rangle\) stays trainable at depth 2, and decays once the circuit is as deep as the number of qubits.
- Depolarizing noise after each layer kills the local gradient at fixed \(n = 6\) as the depth grows.

The notebook stops at 12 qubits, on a statevector and one density matrix. It never builds the portfolio Hamiltonian at \(n \gtrsim 25\), never compiles that Hamiltonian onto the device, and never runs an optimizer there. A dense all-to-all cost layer does not fit the heavy-hex coupling without a depth that the noise section says will erase the gradient. I did not find another encoding that avoids that. That is the gap.

## Layout

```
track_2_optimization/
├── beginner_portfolio_maxcut_qaoa.ipynb   # part 1, QAOA
├── intermediate_portfolio_qml.ipynb       # part 2, quantum ML
├── qml_explanation.md                     # notes for part 2
├── intermediate_qml_slides.pptx           # three slides for part 2
├── advanced_barren_plateaus.ipynb         # part 3, unfinished: a trainability scan
└── challenge_2_tutorial.ipynb             # the Max-Cut QAOA tutorial the track started from
```

The notebooks were run with Qiskit 2.3 and Python 3.14. Re-running the training cells repeats the COBYLA searches. The hardware cell in the quantum-ML notebook only reads the finished job. It does not submit another one. No API token is stored in this repository.
