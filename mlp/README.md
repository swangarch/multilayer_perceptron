# sw-mlp

A NumPy multilayer perceptron for regression and classification, with live training visualization.

Requires Python **3.12**.

```bash
pip install sw-mlp
```

## Example: ReLU regression

Generate noisy 1D data and train a model, using the same setup as the repository's `regre-relu` example. No external dataset or configuration file is needed.

```python
import mlp

x, y = mlp.generate_data_rand(seed=145, number=500, noise_scale=0.02)
model = mlp.MultiLayerPerceptron(
    shape=[1, 64, 32, 1],
    activation_functions=[mlp.relu, mlp.relu, None],
    init_methods=["he", "he", "he"],
)
model.train(
    x, y,
    max_iter=4000,
    learning_rate=0.02,
    batch_size=100,
    threshold=0,
    animation="plot",
)
model.save_plots()

x_test, y_test = mlp.generate_data_rand(seed=123, number=50, noise_scale=0.02)
model.test(x_test, y_test)
```

Run as a Python script with a graphical Matplotlib backend to see live predictions.
Outputs: `params.json`, `predictions.json`, and `visualize/loss.png` in the current directory.

[Source code and documentation](https://github.com/swangarch/multilayer_perceptron)
