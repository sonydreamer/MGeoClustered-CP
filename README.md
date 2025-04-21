# Manifold-Guided Geodesic Clustered Conformal Prediction in Classification
Deep learning models often produce overconfident predictions due to both epistemic and aleatoric uncertainties, which pose significant risks in safety-critical applications. 
To mitigate these risks, conformal prediction (CP) constructs prediction sets with statistical coverage guarantees to quantify uncertainty in deep learning models, without requiring distributional assumptions.
However, most CP approaches operate solely in the output space, overlooking rich information in the feature space.
While Feature Conformal Prediction establishes the first feature-based CP framework by extending CP to feature space, it is primarily designed for regression tasks and exhibits limited applicability to classification tasks.
Additionally, conventional CP approaches rely on uniform calibration across all classes, overlooking class-wise uncertainty heterogeneity and leading to imbalanced coverage across different classes. 
To address these challenges, we propose manifold-guided Geodesic clustered conformal prediction (\textit{MGeoClustered-CP}).
First, we design a novel manifold-guided Geodesic nonconformity score function that not only extends feature-based CP to classification tasks but also accurately quantifies nonconformity between high-dimensional representations by leveraging their Riemannian manifold structures.
Second, we incorporate a fine-grained calibration mechanism into the feature-based CP framework, effectively mitigating coverage disparities across different classes.
Empirical evaluations on four image classification benchmarks demonstrate that \textit{MGeoClustered-CP} constructs more compact prediction sets for classification tasks while achieving balanced coverage across different classes.
In conclusion, our approach contributes to reliable uncertainty quantification for classification models, potentially improving their safety guarantees in critical applications.
