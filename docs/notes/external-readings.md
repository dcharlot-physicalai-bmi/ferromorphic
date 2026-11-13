# External readings, November 2026

These notes record what the primary sources say. They are not an adoption of those methods, and they are not a joint execution with another repository.

## Ferromotion clock tip

Branch `cursor/clock-receipt-v2-epoch-58eb` at `8a2de44a5225b74972528ef2e3fe9c2a5cc4a618` is two commits ahead of `b21ee6d9b2e71ae91a6a5041127f2cb3be290c76` and not behind it. The clock sources, the receiver, the receipt test, and `FERROMORPHIC_CLOCK_REVISION` (`15e219b8f7f1bc76029c359df676cbbcf9817ec0`) are unchanged between those two commits. `FERROMORPHIC_REVISION` remains `8dabe3a45e589908f524b595da09bbadbc8aa103`. The new commits only widen CI push filters to `cursor/**` and replace the hierarchical-quadratic-program pseudoinverse with a scalar Jacobi sweep. `JOINT_LOOP_EXECUTED`, `EPISODE_ENVELOPE_INTEROPERABLE`, and `FERROMORPHIC_BIND_VERIFIED` stay false in that crate. The spike-tick error is named `Unordered` here and `StampMismatch` there. A receipt round trip is not a joint loop.

## arXiv:2610.20147

A. Provata, M. Syroukis, and J. Hizanidis, NCSR Demokritos and FORTH. https://arxiv.org/abs/2610.20147 (14 October 2026). Inspiration only. The abstract and the body do not mention Mackey–Glass. The systems are Lorenz, Rössler, and FitzHugh–Nagumo. What changes relative to a one-shot ridge readout is the training rule: Widrow–Hoff LMS on an augmented state, `w(n+1) = w(n) - r * error * z`, with `r` about `1/N`, leak `0.9`, bias `1`, and Erdős–Rényi or Watts–Strogatz reservoirs. The testing stage feeds the true test sample at every step. It is not a closed-loop autonomous forecast, and it does not specify a NeuroBench Mackey–Glass run. No Rust prototype was added for that protocol. The comment value `40.803375080880265` was not used as a tuning target. That comment is the 14-tau mean, not a tau-30 score.

## arXiv:2610.14454

Phi Hung Nguyen and Daniel Soh, University of Arizona. https://arxiv.org/abs/2610.14454 (10 October 2026). Inspiration only. The classical baseline is specified as a dense echo-state network, five-step Mackey–Glass, Euler step 1, equation `0.2 x(t-17)/(1+x(t-17)^10) - 0.1 x`, `x(0)=1`, 1,000 steps discarded, then 100 washout, 2,000 train, 895 test, ridge `1e-10`, and the 810-point grid in Supplementary Note 14. Table 19 at `m=4` is held-out NRMSE mean `0.2516` and std `0.0418` over 10,000 draws. The delay history and the random generator are not stated. A history that is also 1 is the equilibrium of that ODE (training-target std 0). `tools/qrc_esn_baseline.py` assumes a delay history of 0.9, which gives training-target std `0.2330` against their quoted `0.2331`, and 40 NumPy seeds. That run's held-out mean was `0.2453` with sample std `0.0376`. It is an external range check under that assumption, not their table, and not the NeuroBench autonomous protocol.

## NeurIPS 2025 generalist decoder and FALCON

Joel Ye and coauthors, "A Generalist Intracortical Motor Decoder", NeurIPS 2025. https://proceedings.neurips.cc/paper_files/paper/2025/file/a00000e6a2208172700510bcd69d48e9-Paper-Conference.pdf Inspiration only. The NeurIPS copyright FAQ (https://neurips.cc/FAQ/Copyright) says authors keep copyright and grant NeurIPS a non-exclusive distribution license. That page does not state a CC-BY grant. Code at https://github.com/joel99/ndt3 is MIT. FALCON is Karpowicz et al., bioRxiv https://doi.org/10.1101/2024.09.15.613126, challenge code https://github.com/snel-repo/falcon-challenge (MIT). FALCON M1-A, https://dandiarchive.org/dandiset/000941, is CC-BY-4.0 reach-and-grasp with EMG. The paper's Indy evaluation decodes 2D velocity from M1 on Zenodo https://doi.org/10.5281/zenodo.788569 with 20 ms tokens. That is not the NeuroBench 4 ms, window-50, six-session column-mean R². It was not run here.
