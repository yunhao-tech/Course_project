# Lecture5_tools_independance_test.ipynb
- DoWhy: a Python library that aims to spark causal thinking and analysis. **Should be your first choice when perform causal inference analysis**.

- pycit: Framework for **independence testing and conditional independence testing**, with multiprocessing. Currently **uses mutual information (MI) and conditional mutual information (CMI) as test statistics**

- PyRKHSstats: Provides the Hilbert-Schmidt Independence Criterion **(HSIC) for independence testing**, which is a dependence measure based on **reproducing kernel Hilbert spaces**

- **Causal Discovery Toolbox** (cdt): a toolbox for causal inference in graphs. E.g. provides PC algorithm.

---
# Lecture5_causal_learn.ipynb
- **PC algorithm**: **make use of conditional independence to discover the causal graph model**.

- LiNGAM: Estimation of **Linear, Non-Gaussian Acyclic causal Model** from observed data. It assumes **non-Gaussianity of the noise terms** in the causal model.

---

# Lecture5_Structure_learning_additive_noise.ipynb
Additive noise model (ANM)

Identifiability: if model is allowed in only 1 causal direction

The model is identifiable if:
- The cause-consequence relationship is **non-linear**.
- The cause-consequence relationship is **linear**, and **cause or noise are not normally distributed**. 

Attention: if cause-consequence relationship is linear; cause and noise are both normally distributed, then the model is not identifiable (we could not identifier the cause and consequence).

