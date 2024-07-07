import numpy as np
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adaboost import AdaBoost
from decision_stump import DecisionStump


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    model = AdaBoost(DecisionStump, n_learners)
    model.fit(train_X, train_y)

    # We save the training and test errors at each iteration to plot them later
    train_error = [model.partial_loss(train_X, train_y, t) for t in range(1, n_learners + 1)]
    test_error = [model.partial_loss(test_X, test_y, t) for t in range(1, n_learners + 1)]

    # Plot the training and test errors
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(1, n_learners + 1), y=train_error, mode='lines', name='Train error'))
    fig.add_trace(go.Scatter(x=np.arange(1, n_learners + 1), y=test_error, mode='lines', name='Test error'))
    fig.update_layout(title=f'AdaBoost Error For Number Of Learners - Noise Ratio: {noise}',
                      xaxis_title='Number of learners',
                      yaxis_title='Misclassification error')
    fig.show()
    # fig.write_image(f'adaboost_noise_{noise}.png')

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{t} Iterations" for t in T])

    for i, t in enumerate(T):
        # Create decision surface
        contour = decision_surface(lambda x: model.partial_predict(x, t), lims[0], lims[1], showscale=False)
        fig.add_trace(contour, row=i // 2 + 1, col=i % 2 + 1)

        # Add test points to the plot
        fig.add_trace(go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode='markers',
                                 marker=dict(color=test_y, colorscale=['red', 'blue'], showscale=False,
                                             line=dict(color='black', width=1)),
                                 showlegend=False), row=i // 2 + 1, col=i % 2 + 1)

    fig.update_layout(title=f'AdaBoost Decision Boundaries for Different Iterations - Noise Ratio: {noise}',
                      margin=dict(t=100))
    fig.show()
    # fig.write_image(f'adaboost_noise_{noise}_decision_boundaries.png')

    # Question 3: Decision surface of best performing ensemble

    # Find the ensemble size with the lowest test error
    best_iteration = np.argmin(test_error) + 1  # +1 because iterations are 1-indexed

    # Plot decision surface and test points for the best iteration
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    best_preds = model.partial_predict(test_X, best_iteration)

    fig = make_subplots(rows=1, cols=1)

    # Create decision surface
    contour = decision_surface(lambda x: model.partial_predict(x, best_iteration), lims[0], lims[1], showscale=False)
    fig.add_trace(contour)

    # Add test points to the plot
    fig.add_trace(go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode='markers',
                             marker=dict(color=test_y, colorscale=['red', 'blue'], showscale=False,
                                         line=dict(color='black', width=1)),
                             showlegend=False))

    # Update layout with title
    fig.update_layout(
        title=f'Best AdaBoost Decision Boundary for {best_iteration} Learners - Accuracy: {1 - test_error[best_iteration - 1]:.2f}, Noise Ratio: {noise}')

    fig.show()
    # fig.write_image(f'adaboost_noise_{noise}_best_decision_boundary.png')


    # Question 4: Decision surface with weighted samples

    # Get weights from the last iteration
    D = model.D_

    # Normalize and transform weights for visualization
    D_normalized = 25 * D / D.max()

    fig = make_subplots(rows=1, cols=1)

    # Create decision surface using the full ensemble
    contour = decision_surface(model.predict, lims[0], lims[1],density=60, showscale=False)
    fig.add_trace(contour)

    # Add training points to the plot with weighted size and colored by label
    fig.add_trace(go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode='markers',
                             marker=dict(color=train_y,
                                         size=D_normalized,
                                         symbol=np.where(train_y == 1, "circle", "x")),
                             showlegend=False))

    # Update layout with title
    fig.update_layout(width=500, height=500,
                         title=f"Final AdaBoost Sample Distribution - Noise Ratio: {noise}")
    fig.show()
    # fig.write_image(f"adaboost_{noise}_weighted_samples.png")

if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
