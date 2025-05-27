# <p align="center">Theoretical Bounds for Robust Model Evaluation over Federated Networks</p>
## Introduction
This repository contains the code for the paper [Theoretical Bounds for Robust Model Evaluation over Federated Networks]
## Abstract
Consider a network of clients with private, non-IID local datasets governed by an unknown meta-distribution. A central server aims to evaluate the average performance of a given ML model, not only on this network (standard beta or A/B testing) but also on all possible unseen networks that are meta-distributionally similar to it, as measured by either $f$-divergence or Wasserstein distance. To this end, we propose a novel robust optimization framework that can be implemented in a private and federated manner with at most polynomial time and query complexity. Specifically, we introduce a private server-side aggregation technique for local adversarial risks, which provably enables a global robust evaluation of the model. We also establish asymptotically minimax-optimal bounds for the risk average and CDF, with vanishing generalization gaps as the source network size $K$ grows and the minimum local dataset size exceeds $\mathcal{O}\left(\log K\right)$. Empirical results further validate the effectiveness of our bounds in real-world tasks.

## Results
###  Client Generation and Risk CDF Certificates for Unseen Clients
<p align="center">
  <img src="images/single.PNG" alt="Alt Text" width="700">
</p>

### Certificates for f-Divergence Meta-Distributional Shifts
<p align="center">
  <img src="images/Meta.png" alt="Alt Text" width="700">
</p>

<p align="center">
  <img src="images/two-meta.PNG" alt="Alt Text" width="700">
</p>
