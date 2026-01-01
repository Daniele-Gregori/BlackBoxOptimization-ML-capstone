# Black Box Optimization Capstone Project

## 1. Project Overview

The Black Box Optimization (BBO) Capstone Project focuses on optimizing complex mathematical functions through the application of Bayesian optimization techniques. The primary purpose of this project is to develop robust machine learning models that can effectively explore input spaces to identify optimal solutions, maximizing or minimizing specific target functions.

The overall goal of the BBO capstone project is to create an effective optimization framework that balances exploration and exploitation, enabling the discovery of global maxima with minimal computational resources. This project holds significant relevance in real-world machine learning as it addresses complex optimization problems commonly encountered in various domains such as engineering design, finance optimization, and hyperparameter tuning in machine learning models. The high-level idea is to systematically evaluate and refine approaches to maximize function outputs in uncertain and multidimensional spaces.

Engaging in this BBO capstone project supports my current and future career aspirations by providing hands-on experience with advanced machine learning techniques, enhancing my problem-solving abilities in real-world applications, and preparing me for roles in data science and AI development.


## 2: Inputs and Outputs

The model receives and returns the following:

- **Inputs**:
  - **Query Format**: A structured dictionary containing the function index and input parameters in a list format.
  - **Dimensions**: The input space can consist of multiple dimensions, ranging from 2 to 8, depending on the specific function being analyzed.
  - **Constraints**: The input parameters are restricted to specific ranges, and after standardization these are the unit hypercubes.

- **Expected Output**:
  - **Response Value**: The predicted maximum value (or minimum, depending on the optimization goal) of the function based on the given input parameters.
  - **Performance Signal**: An uncertainty measure associated with the prediction, reflecting the confidence of the model in its output.

Example input format:
```python
{
    function_index: 2,
    parameters: [0.5, 0.7, 0.2]
}
```

Expected output format:
```python
{
    predicted_value: 0.85,
    uncertainty: 0.02
}
```

## 3: Challenge Objectives

Within the BBO Capstone Project, the primary objective is to maximize the selected functions by effectively exploring the input space and identifying their maxima using a blend of Machine Learning (ML) methods.

The goal is to maximize the function outputs while adhering to the following constraints and limitations:
- **Number of Queries**: The optimization process is typically limited to a few tens of queries, which specifies the number of evaluations of the functions.
- **Response Delay**: Given the extensive computational tasks involved, responses may take varying amounts of time, depending on the dimensionality of the functions being evaluated.
- **Unknown Function Structure**: The functions can exhibit complex behaviors and non-linear relationships, which complicates modeling and optimization efforts. It may lead to slow convergence at times, challenging the efficiency of the optimization process.


## 4: Technical Approach

Throughout the course of my BBO capstone project, I have developed a systematic approach to optimizing complex functions, which evolved across three query submissions. This section serves as a living record of the strategies and methodologies employed, and it will be updated as my approach continues to evolve.

### Query Submissions and Strategies

1. **Round One: Initial Exploration and Gaussian Processes**
   - In the initial submission, I utilized Gaussian Processes (GP) for function approximation. This approach allowed me to predict the expected value and uncertainty associated with function outputs based on a limited dataset provided for each function.
   - I created a uniform grid across the input space with varying subdivisions for different functions, allowing thorough exploration of the domain. This brute-force method enabled me to evaluate many points and store results efficiently.
   - I applied the Upper Confidence Bound (UCB) acquisition function where the exploration-exploitation balance was initially set to balance exploitation and exploration, in a completely agnostic way.

2. **Round Two: Increased Exploration with Modified UCB**
   - Based on the findings from the first round, I shifted my focus toward exploration in the second submission. After observing stagnation in identifying new maxima, I adjusted the beta parameter within the UCB function to enhance exploration.
   - The beta value was heuristically set to a higher multiplier (around 4) to actively encourage the model to explore more diverse regions of the input space, particularly where new maxima might remain undiscovered.

3. **Round Three: SVM Integration and Remote Batch Computation**
   - In this round, I integrated Support Vector Machines (SVM) into my methodology. Although SVM is typically used for classification, I exploited it to define promising regions of the input space. Using SVM, I categorized outputs into quartiles to focus exploration on regions likely to yield higher values.
   - Furthermore, I utilized remote batch computation to improve efficiency, significantly decreasing computation times when executing extensive grid searches. This move was particularly beneficial for high-dimensional functions.
   - My exploration strategy was refined using insights from the SVM model, which helped guide the selection of query points, allowing for a more thoughtful balance between exploration and exploitation (determined by requiring both by the convergence of the UCB decision for increasing beta parameter and agreement with the SVM decision).

### Machine Learning Methods and Heuristics

Throughout the project, I primarily employed the following methodologies:
- **Gaussian Processes** for modeling unknown functions: GP was used to provide predictions and uncertainty estimates for the function outputs based on input parameters.
- **Support Vector Machines** as a heuristic to classify promising regions of input space, directing exploration towards areas that may yield better outcomes.
- Exploration strategies were employed using the UCB acquisition function to balance exploration and exploitation effectively, continuously refining this balance as new insights were gained.

### Balancing Exploration and Exploitation

My approach to balancing exploration and exploitation involves a dynamic adaptation of the beta parameter in the UCB acquisition function. By using insights from SVM's categorization, I strategically increase the beta value to promote exploration in potentially fruitful regions while maintaining an awareness of previously identified maxima.

What makes my approach unique is this integration of diverse methods - leveraging both probabilistic modeling (GP) and classification techniques (SVM). This combination allows for a nuanced exploration strategy that can adapt and respond to the inherent complexities present in the functions being optimized.

