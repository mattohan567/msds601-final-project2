import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def generate_data(n_points=100):
    """
    Generates a random dataset with a linear relationship and some noise.
    """
    np.random.seed(42)  # for reproducible results
    x = np.random.rand(n_points) * 10  # Random values from 0 to 10
    y = 3.5 * x + np.random.randn(n_points) * 3  # Linear relationship with noise
    return x, y
    

def plot_k_folds(n_points, k):
    """
    Creates a plot showing the training and testing splits for K-fold cross-validation with clear borders.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    fold_size = n_points // k
    remainder = n_points % k
    
    legend_handles = [Patch(facecolor='skyblue', label='Training Data'),
                      Patch(facecolor='salmon', label='Testing Data')]

    start = 0
    for i in range(k):
        end = start + fold_size + (1 if i < remainder else 0)

        # Plot each fold with a border
        # Testing set
        ax.add_patch(plt.Rectangle((start, i), end - start, 1, facecolor='salmon', edgecolor='black', linewidth=1))
        # Training set before the testing set
        if start > 0:
            ax.add_patch(plt.Rectangle((0, i), start, 1, facecolor='skyblue', edgecolor='black', linewidth=1))
        # Training set after the testing set
        if end < n_points:
            ax.add_patch(plt.Rectangle((end, i), n_points - end, 1, facecolor='skyblue', edgecolor='black', linewidth=1))

        start = end
    
    ax.set_ylim(0, k)
    ax.set_xlim(0, n_points)
    if k <= 20:
        ax.set_yticks(range(k))
        ax.set_yticklabels([f'Fold {i+1}' for i in range(k)])
    else: 
        ax.set_yticks(range(0, k+1, 5))
        ax.set_yticklabels([f'Fold {i+1}' for i in range(0, k+1, 5)])
    ax.set_title(f'{k}-Fold Cross-Validation')
    ax.legend(handles=legend_handles, loc='upper right')
    ax.set_xlabel('Data Points')
    # ax.set_ylabel('Fold Number')
    ax.invert_yaxis()
    st.pyplot(fig)

criterias = ['neg_mean_squared_error', 'r2']
def perform_cross_validation(x, y, k):
    """
    Performs K-fold cross-validation on a simple linear regression model.
    """
    model = LinearRegression()
    scores = {}
    for criteria in criterias:
        scores[criteria] = cross_val_score(model, x.reshape(-1, 1), y, cv=k, scoring=criteria)
    return scores

def main():
    st.title("K-Fold Cross-Validation in Simple Linear Regression")

    # Extensive introduction and detailed explanation
    st.markdown("""
    ## Cross-Validation in Linear Regression: An In-depth Exploration
    
    ### Introduction
    Cross-validation stands as a cornerstone technique in the realm of statistical analysis and predictive modeling. This method evaluates the efficacy of a model by partitioning the original dataset into complementary subsets, where one subset is used to train the model and the other to test its performance. This validation approach is essential for verifying the generalizability and robustness of predictive models, ensuring they perform consistently across various sets of data.
    
    ### Why Embrace Cross-Validation?
    - **Enhanced Model Evaluation**:
      - Cross-validation provides a more comprehensive assessment of a model’s performance compared to a simple train/test split. It ensures that the model’s effectiveness is tested across all data points, mitigating the risk of anomalies in any single train-test partition.
    - **Bias-Variance Tradeoff**:
      - It addresses the critical challenge in model training: balancing complexity and accuracy. Cross-validation helps in minimizing overfitting, ensuring the model performs well on any unseen dataset, not just the data on which it was trained.
    - **Model Selection**:
      - Through its iterative process, cross-validation aids in selecting the model that best handles new, unseen data. It allows for comparing the performance of different models on the same dataset, facilitating informed decision-making about the optimal model choice.

    ### Types of Cross-Validation
    - **K-Fold Cross-Validation**:
      - This method involves dividing the data into ‘K’ subsets or folds. Each subset serves as a test set once, while the remaining subsets are used as the training set. This cycle repeats ‘K’ times, with each of the ‘K’ subsets used exactly once as the test set. The average of the results from all ‘K’ folds provides a comprehensive performance estimate.
    - **Stratified K-Fold Cross-Validation**:
      - An enhancement of the K-fold method, stratified cross-validation ensures that each fold is a good representative of the whole by having approximately the same percentage of samples of each target class as the complete set.
    - **Leave-One-Out (LOO) Cross-Validation**:
      - In LOO, each observation is used as a separate test set, while the rest serve as the training set. This method is particularly beneficial when the dataset is small, as it maximizes the training data. However, it can be computationally expensive.

    ### Advantages of Cross-Validation
    - **Robustness**:
      - By using multiple subsets of the data for training and testing, cross-validation provides a robust measure of model’s performance, reducing the variability and enhancing the reliability of the model assessment.
    - **Resource Efficiency**:
      - It makes efficient use of available data, ensuring every observation is used for both training and validation, thus maximizing data utility and avoiding wastage.
    - **Versatility and Flexibility**:
      - Cross-validation can be applied to any model, whether it’s linear, logistic, or any other type of regression or classification model. This flexibility makes it an indispensable tool across various domains of data analysis.
      
    ### Implementing Cross-Validation
    - **Practical Considerations**:
      - The choice of the number of folds in K-fold cross-validation typically balances between computational cost and model performance reliability. A common practice is to use 10 folds, which has been empirically shown to yield a reliable estimate of model performance.
    - **Software and Tools**:
      - Several statistical software packages and programming libraries support cross-validation. Python’s Scikit-learn, for instance, offers extensive support for implementing various forms of cross-validation, with built-in functions that simplify the application of these techniques.
      """)
    
        # Data generation and plotting
    x, y = generate_data()
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='blue', alpha=0.5)
    ax.set_title("Generated Dataset")
    ax.set_xlabel("Predictor")
    ax.set_ylabel("Response")
    ax.grid(True)
    st.pyplot(fig)

    # Slider for choosing K
    k = st.slider("Choose the number of folds for K-fold Cross-Validation", min_value=2, max_value=20, value=5, step=1)

    # Dynamic K-fold visualization
    plot_k_folds(len(x), k)
    
    # Automatically update cross-validation results
    for criteria in criterias:
        scores = perform_cross_validation(x, y, k)
        # st.write(f"{criteria} scores for each fold: {scores[criteria]}")
        st.write(f"Average {criteria} score: **{np.mean(scores[criteria])}**")

    st.markdown("""   
    ### Steps of Cross-Validation: A Step-by-Step Guide
    **Step 1: Prepare Your Data**
    - Data Cleaning: Begin by cleaning your dataset. This includes handling missing values, removing duplicates, and ensuring data consistency.
    - Feature Selection: Identify which features (variables) in the dataset will be included in the model. This might involve exploratory data analysis and feature engineering to create new variables that can improve model performance.
    
    **Step 2: Split the Data into K Folds**
    - Divide Equally: Partition the data into ‘K’ equal or nearly equal subsets. The value of ‘K’ is chosen based on the size of the dataset and the balance between computational efficiency and model evaluation precision.
    - Stratification (Optional): For classification problems, ensure each fold represents the overall composition of the output variable through stratification. This means each fold contains approximately the same percentage of samples of each target class as the original dataset.
    
    **Step 3: Execute the Cross-Validation**
    - Iterate Through Folds: For each of the ‘K’ folds:
    - Training Set Creation: Use ‘K-1’ folds as the training set.
    - Testing Set Selection: Use the remaining fold as the testing set.
    - Model Training: Train the model on the ‘K-1’ folds designated as the training set.
    - Model Testing: Evaluate the model using the fold reserved as the testing set.
    
    **Step 4: Evaluate and Aggregate the Results**
    - Performance Metrics: After testing the model on each fold, record the performance metrics such as accuracy, precision, recall, F1-score, or mean squared error, depending on the type of model (classification or regression).
        - When using the cross_val_score function from Scikit-learn with a model such as LinearRegression, you can specify different scoring parameters depending on what aspect of the model’s performance you wish to evaluate. For regression models like LinearRegression, common scoring options include:
            - 'r2': The coefficient of determination  R^2  score. It provides an indication of goodness of fit and the percentage of the response variable variation that is explained by a linear model.  R^2  is probably the most commonly used metric for evaluating the performance of regression models.
            - 'neg_mean_squared_error': Negative mean squared error. It is a risk metric corresponding to the expected value of the squared (quadratic) error or loss. The higher this number (less negative), the better, as it indicates a model with lesser errors on average. The ‘neg_’ prefix indicates that a higher score is better and allows for consistency when comparing with other metrics (which assume that a higher score is better).
            - 'neg_mean_absolute_error': Negative mean absolute error. This measures the average magnitude of the errors in a set of predictions, without considering their direction (i.e., it averages the absolute values of the errors).
            - 'neg_root_mean_squared_error': Negative root mean squared error. This is the square root of the mean of the squared errors. RMSE is sensitive to outliers and can give a clearer idea of the model performance when large errors are particularly undesirable.
            - 'neg_median_absolute_error': Negative median absolute error. This metric is particularly interesting because it is robust to outliers. The median of all absolute differences between the targets and the predictions can be a more representative metric of the typical prediction error than the mean error.
            - 'max_error': The maximum residual error. A metric that captures the worst case error between the predicted value and the true value. In a forecasting context, this can provide you with worst-case scenario information.
            - 'explained_variance': Measures the proportion to which a mathematical model accounts for the variation (dispersion) of a given dataset. Unlike  R^2 , this metric does not necessarily penalize excessive complexity in the model, making it useful when you want to capture the variance explained by the model but are less concerned about the model’s simplicity.
    - Aggregate Outcomes: Calculate the average of the performance metrics across all ‘K’ folds to obtain an overall estimate of the model’s effectiveness.
    
    **Step 5: Model Selection and Adjustment**
    - Compare Models: If multiple models are being tested, compare the aggregated results from the cross-validation process to select the best-performing model.
    - Tune Hyperparameters: Based on the cross-validation results, adjust model parameters to refine its performance. This might involve tuning hyperparameters like learning rate, the number of trees in a random forest, or the number of hidden layers in a neural network.
    
    **Step 6: Final Model Training and Validation**
    - Train Final Model: Once the best model and parameters are identified, train this final model on the entire dataset to maximize its learning.
    - External Validation (Optional): If additional unseen data is available, validate the final model on this external dataset to further confirm its generalizability and robustness.
    """)


if __name__ == "__main__":
    main()