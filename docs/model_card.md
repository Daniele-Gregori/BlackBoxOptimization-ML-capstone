# Model Card: BBO Capstone Optimization Approach

## Overview

- **Name:** Hybrid Bayesian Surrogate Optimization with Adaptive Strategy Selection
- **Type:** Sequential model-based black-box optimization
- **Version:** 10.0 (intermediate round)
- **Framework:** Python with scikit-learn, PyTorch, and custom implementations
- **Developer:** Single researcher, academic capstone project
- **Date:** Ten-round development spanning approximately ten weeks

## Intended Use

**Suitable tasks:**
- Black-box optimization of continuous functions with limited evaluation budgets
- Low-dimensional to moderate-dimensional problems (2-8 dimensions)
- Scenarios where function evaluations are expensive and each query must be carefully chosen
- Problems where uncertainty quantification is valuable for guiding exploration

**Use cases to avoid:**
- High-dimensional optimization beyond 10 dimensions, where the grid-based components would be computationally infeasible
- Problems requiring real-time or low-latency decisions, as the GP fitting and grid evaluation pipeline requires minutes to hours
- Discrete or combinatorial optimization problems, as the approach assumes continuous input spaces
- Problems with very large evaluation budgets where simpler methods like random search or evolutionary algorithms may be more efficient
- Safety-critical applications where the lack of formal convergence guarantees would be unacceptable

## Details

### Strategy Evolution Across Ten Rounds

**Phase 1 -- Baseline Establishment (Rounds 1-2)**

The initial approach used Gaussian Process regression as a surrogate model, evaluated over uniform grids spanning the input domain. The UCB acquisition function balanced exploration and exploitation, with beta initially set as the ratio of mean predicted values to standard deviations. Grid subdivisions ranged from 5 to 15 per dimension, producing up to 16 million evaluation points for higher-dimensional functions. Parallel computation across 8-24 CPU cores was exploited. In round 2, beta was increased to approximately 4 to emphasize exploration after observing that no new maxima were being discovered.

Key techniques:
- Gaussian Process regression with default kernel
- UCB acquisition function with heuristic beta
- Brute-force grid evaluation with parallel computation
- Manual handling of Function 1 due to its anomalous data distribution

**Phase 2 - Model Diversification (Rounds 3-6)**

This phase introduced multiple complementary modeling approaches. In round 3, Support Vector Machine classification was used to identify promising regions by labeling the upper quartile of outputs as positive, with the SVM decision boundary informing where to push UCB exploration. Rounds 4 through 6 introduced a Bayesian Neural Network built from scratch in PyTorch, with three Bayesian linear layers of dimensions 32, 16, and 8. This provided an alternative source of predictions and uncertainty estimates. Critically, gradient computation was implemented to assess feature sensitivity and enable gradient ascent from best known points.

Key techniques:
- SVM classification with C=1 for region identification
- Bayesian Neural Network with KL-divergence regularized loss
- Monte Carlo forward passes for predictive uncertainty
- Gradient-based sensitivity analysis per input dimension
- 3D visualization of pairwise dimension slices to characterize surface types
- Remote batch computation on 192-core machine

**Phase 3 - Systematic Refinement (Rounds 7-8)**

Round 7 represented the most impactful methodological advance. Systematic hyperparameter tuning tested 48 configurations comprising 8 kernel types and 6 alpha values, evaluated with 3-fold cross-validation. Key findings included Matern 5/2 performing best on half the functions, ARD improving RMSE by 40-60% on high-dimensional functions, and optimal alpha values between 1e-5 and 1e-4. Round 8 explored LLM-assisted code generation through the Anthropic API, testing different temperature and sampling parameters.

Key techniques:
- Grid search over kernel and alpha hyperparameters
- 3-fold cross-validation for robust evaluation
- Automatic Relevance Determination for feature importance
- LLM code generation with systematic parameter variation

**Phase 4 - Synthesis (Rounds 9-10)**

The final rounds combined the strongest elements from all previous phases into a unified pipeline. The tuned GP with function-specific kernel and alpha served as the core surrogate. ARD length scales identified irrelevant dimensions, which were fixed at their current best values to reduce the effective search space. Gradient ascent from the best known point in the reduced space proposed candidate queries, which were then validated against UCB predictions. The strategy shifted decisively from exploration toward exploitation for functions with well-characterized landscapes.

Key techniques:
- Function-specific GP configurations from round 7 tuning
- ARD-based dimension reduction for functions 7 and 8
- Gradient ascent in reduced dimensional space
- UCB validation of gradient-suggested candidates
- Surface-type-specific exploration versus exploitation balance

### Decision-Making Process

The approach makes decisions through a layered pipeline. First, the GP surrogate is fitted to all available data using the best hyperparameters identified through cross-validation. Second, ARD length scales are extracted to determine feature relevance, and dimensions with very large length scales are fixed. Third, gradient ascent from the current best point proposes a candidate in the reduced space. Fourth, UCB with moderate beta evaluates whether the gradient candidate or an alternative point offers better expected value adjusted for uncertainty. The final query is selected as the point with the highest acquisition function value among the candidates considered.

## Performance

### Results Summary

Performance is measured by the **best function value achieved** across all rounds for each function and by the **RMSE of the GP surrogate** as assessed through cross-validation.

| Function | Dimensions | Landscape Type | Best Kernel | ARD Benefit | Convergence |
|----------|-----------|----------------|-------------|-------------|-------------|
| 1 | 2 | Anomalous single point | Manual handling | N/A | Limited by data |
| 2 | 2 | Near-linear | Matern 5/2 | Minimal | Good |
| 3 | 3 | Single peak | Matern 5/2 | Moderate | Good |
| 4 | 4 | Mixed extrema | Matern 5/2 | Moderate | Moderate |
| 5 | 4 | Oscillatory | Varies | Moderate | Slow |
| 6 | 5 | Mixed | Matern 5/2 | Significant | Moderate |
| 7 | 6 | Complex | Matern 5/2 + ARD | 40-60% RMSE improvement | Slow |
| 8 | 8 | Complex | Matern 5/2 + ARD | 40-60% RMSE improvement | Very slow |

### Key Metrics

- **RMSE improvement from hyperparameter tuning:** 19% average across functions compared to default configuration
- **RMSE improvement from ARD on high-dimensional functions:** 40-60%
- **Optimal regularization range:** alpha between 1e-5 and 1e-4
- **Grid search diminishing returns threshold:** approximately 10 subdivisions per dimension beyond which marginal gains were negligible

### Observed Patterns

Lower-dimensional functions (1-4) showed reasonable convergence with the GP surrogate providing reliable predictions near sampled regions. Higher-dimensional functions (5-8) exhibited persistent underfitting due to data sparsity, with convergence remaining slow despite methodological improvements. Functions with oscillatory behavior were the most challenging across all rounds.

## Assumptions and Limitations

### Key Assumptions

1. **Smoothness assumption:** The Matern 5/2 kernel assumes the target functions are twice differentiable. If functions contain discontinuities or sharp transitions in unsampled regions, the GP will produce misleadingly smooth interpolations that could direct queries away from true optima.

2. **Stationarity assumption:** The GP assumes the covariance structure is uniform across the input space. Functions with varying complexity in different regions, such as smooth in one area and oscillatory in another, violate this assumption.

3. **Feature relevance stability:** ARD length scales estimated from 19 points are treated as reliable indicators of feature importance. A dimension deemed irrelevant and subsequently fixed could actually be important in unsampled regions, creating an irreversible strategic commitment.

4. **Local optimality of gradient ascent:** Gradient ascent from the best known point assumes the true optimum is reachable from that starting location. Multiple disconnected peaks or rugged landscapes could cause this to converge to a local optimum.

### Limitations

- **Data sparsity is the fundamental bottleneck.** With 19 points in up to 8 dimensions, no surrogate model can reliably characterize the function landscape. This limitation cannot be overcome by model sophistication alone.
- **Sampling bias compounds over rounds.** Model-driven queries cluster in regions identified as promising, potentially missing global optima in unexplored areas. This bias is self-reinforcing as each new point further concentrates the model's confidence in already-sampled regions.
- **Computational cost of grid evaluation scales exponentially** with dimensionality, making exhaustive search infeasible beyond approximately 6 dimensions even with parallel computation.
- **No formal convergence guarantees** exist for the heuristic combination of GP, gradient ascent, and UCB used in the final strategy.
- **The BNN and SVM components were ultimately superseded** by the tuned GP, meaning significant development effort did not directly contribute to final performance.

### Failure Modes

- The approach will fail if the true optimum lies in a region that was never sampled and that the GP predicts with high confidence to be low-valued
- ARD-based dimension pruning will fail if important dimensions show little variation in the sparse training data by coincidence
- Gradient ascent will fail on multimodal landscapes where the current best point sits in the basin of a suboptimal peak

## Ethical Considerations

### Transparency and Reproducibility

This model card and its accompanying datasheet are designed to make the entire optimization process transparent and reproducible. Every strategic decision, from kernel selection to dimension pruning, is documented with its rationale and the evidence that supported it. The evolution of the approach across ten rounds is recorded, including unsuccessful directions such as the BNN development that ultimately did not improve optimization performance. This honest documentation of both successes and dead ends supports scientific integrity.

A learner with access to the initial provided data points, the submission history, and this documentation could reproduce the approach and understand why each decision was made. The primary barrier to exact reproduction is the absence of fixed random seeds in some computational steps and the informal nature of certain judgment calls, such as the threshold for declaring a dimension irrelevant based on ARD length scales.

### Real-World Adaptation

The transparency of assumptions documented above is critical for anyone adapting this approach to real-world problems. In applied settings such as drug discovery, materials science, or engineering design, the consequences of a flawed assumption like incorrect dimension pruning could be significant. Practitioners should validate ARD-based feature selection through domain expertise rather than relying solely on statistical estimates from sparse data.

The approach's reliance on computational resources, including multi-core parallel processing and large intermediate file storage, should be considered when evaluating its feasibility for resource-constrained settings. The shift from brute-force grid search to targeted gradient-based methods in later rounds represents a more sustainable and transferable paradigm.

### Broader Impact

This project involves only mathematical function evaluations and poses no direct risk of societal harm. However, the methodological lessons about operating under uncertainty with limited data are broadly applicable to domains where decisions must be made with incomplete information. The documented tension between exploration and exploitation, and between model sophistication and practical performance, reflects challenges that arise throughout applied machine learning and data science practice.