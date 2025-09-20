"# Math for AI: The Mathematical Foundations of Artificial Intelligence

> *"Mathematics is the language with which God has written the universe... and artificial intelligence is our attempt to decode it."*

Welcome to my **Math for AI** project - my personal journey learning the mathematical foundations that power modern artificial intelligence and machine learning. I created this repository while studying these topics to help transform abstract mathematical concepts into visual, intuitive, and practical understanding through Jupyter notebooks.

## Why Mathematics is the Foundation of AI

As I began my AI learning journey, I quickly realized that Artificial Intelligence isn't magic - it's mathematics in action! I've documented my understanding of how every breakthrough in AI, from neural networks to transformers, is built upon fundamental mathematical principles:

- **Linear Algebra**: Powers data representation, transformations, and neural network computations
- **Calculus**: Enables optimization through gradients and backpropagation
- **Probability & Statistics**: Forms the backbone of uncertainty, inference, and learning
- **Optimization**: Drives the learning process in all ML algorithms

This collection of notebooks represents my attempt to bridge the gap between mathematical theory and AI implementation, exploring not just *what* the math is, but *how* it works and *why* it matters.

## Table of Contents

### [01_Linear_Algebra](./01_Linear_Algebra/)
My exploration of data representation and neural networks
- **[1_vectors_and_matrices.ipynb](./01_Linear_Algebra/1_vectors_and_matrices.ipynb)** - Visual vector operations, matrix fundamentals, and geometric intuition
- **[2_matrix_multiplication_visualization.ipynb](./01_Linear_Algebra/2_matrix_multiplication_visualization.ipynb)** - Interactive matrix multiplication and neural network connections
- **[3_eigenvalues_eigenvectors_pca.ipynb](./01_Linear_Algebra/3_eigenvalues_eigenvectors_pca.ipynb)** - Eigendecomposition, PCA derivation, and MNIST dimensionality reduction

### [02_Calculus](./02_Calculus/)
Learning the mathematics of optimization and learning
- **[1_derivatives_and_gradients.ipynb](./02_Calculus/1_derivatives_and_gradients.ipynb)** - Step-by-step derivatives with SymPy and geometric interpretation
- **[2_chain_rule_backprop.ipynb](./02_Calculus/2_chain_rule_backprop.ipynb)** - Chain rule fundamentals and neural network backpropagation
- **[3_multivariable_gradients.ipynb](./02_Calculus/3_multivariable_gradients.ipynb)** - 3D gradient visualization and optimization landscapes

### [03_Probability_Statistics](./03_Probability_Statistics/)
My study of uncertainty, inference, and probabilistic reasoning
- **[1_random_variables_distributions.ipynb](./03_Probability_Statistics/1_random_variables_distributions.ipynb)** - Monte Carlo simulations and distribution exploration
- **[2_bayes_theorem_and_inference.ipynb](./03_Probability_Statistics/2_bayes_theorem_and_inference.ipynb)** - Bayes' theorem with spam classification example
- **[3_markov_chains_and_probabilistic_models.ipynb](./03_Probability_Statistics/3_markov_chains_and_probabilistic_models.ipynb)** - Random walks and their applications in AI

### [04_Optimization](./04_Optimization/)
My deep dive into the algorithms that make learning possible
- **[1_gradient_descent_visualization.ipynb](./04_Optimization/1_gradient_descent_visualization.ipynb)** - Animated gradient descent on various function landscapes
- **[2_stochastic_gd_and_variants.ipynb](./04_Optimization/2_stochastic_gd_and_variants.ipynb)** - SGD, Momentum, Adam comparison with visual learning curves
- **[3_convex_vs_nonconvex.ipynb](./04_Optimization/3_convex_vs_nonconvex.ipynb)** - Understanding optimization challenges and solutions

### [05_Applications_in_ML](./05_Applications_in_ML/)
Connecting theory to practice with real machine learning implementations
- **[1_linear_regression_math_vs_code.ipynb](./05_Applications_in_ML/1_linear_regression_math_vs_code.ipynb)** - Mathematical derivation alongside practical implementation
- **[2_logistic_regression_derivation.ipynb](./05_Applications_in_ML/2_logistic_regression_derivation.ipynb)** - From maximum likelihood to sigmoid functions
- **[3_backpropagation_demo.ipynb](./05_Applications_in_ML/3_backpropagation_demo.ipynb)** - Mathematical derivation and step-by-step implementation
- **[4_pca_and_dimensionality_reduction.ipynb](./05_Applications_in_ML/4_pca_and_dimensionality_reduction.ipynb)** - PCA, t-SNE mathematical foundations and applications

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Lab or Jupyter Notebook
- Basic understanding of Python programming

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sat-yam-ln2/Math-for-AI.git
   cd Math-for-AI
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv math_ai_env
   
   # On Windows
   math_ai_env\Scripts\activate
   
   # On macOS/Linux
   source math_ai_env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Lab:**
   ```bash
   jupyter lab
   ```

5. **Start exploring!** Begin with any notebook that interests you - I designed them to be both standalone and interconnected.

## What Makes This Project Special

### Interactive Visualizations
As I learned, I created visualizations to help myself understand concepts better:
- **Animated plots** showing algorithmic processes
- **Interactive widgets** for parameter exploration
- **3D visualizations** for multidimensional concepts
- **Step-by-step breakdowns** of complex derivations

### Theory + Practice Approach
I structured each notebook consistently to help my learning:
1. **Mathematical Theory** - LaTeX equations with intuitive explanations
2. **Visual Demonstrations** - Plots and animations to build intuition
3. **Code Implementation** - NumPy/PyTorch examples showing practical usage
4. **Real Applications** - Connections to actual AI/ML problems

### Modular Learning
I designed this for my own nonlinear learning style:
- **Self-contained notebooks** - Jump to any topic that interests you
- **Progressive complexity** - Build from fundamentals to advanced concepts
- **Cross-references** - See how different mathematical areas connect

## Technologies Used

During this learning journey, I used:
- **Python** - Primary programming language
- **NumPy & SciPy** - Numerical computing
- **Matplotlib & Plotly** - Static and interactive visualizations
- **SymPy** - Symbolic mathematics
- **Scikit-learn** - Machine learning implementations
- **PyTorch** - Deep learning examples
- **Jupyter** - Interactive development environment

## Sharing and Using This Work

If you find something that could be improved:
- Fix typos or improve explanations
- Add new visualizations or examples
- Suggest new topics or notebooks
- Report bugs or issues

Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- I was inspired by the amazing work of educators like Grant Sanderson (3Blue1Brown)
- This work builds on the open-source scientific Python community
- I created this because I believe mathematics should be accessible and visual

---

If you find this repository helpful for your own AI/ML journey, please feel free to use it and share it with others.

**Happy Learning!**

*I created this project to help myself not just use AI tools, but to understand the beautiful mathematics that makes them work.*" 
