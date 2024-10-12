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
    st.title("Predicting Car Prices 🚗: A Journey Through Cross-Validation and Model Selection in Regression")

    st.markdown("""
    ## Introduction

    When buying a used car, we're often swamped with questions: *How much should this car cost?*  
    Are certain features driving the price up or down? What if we could use data to predict car prices more accurately?

    This blog post is your roadmap to creating a used car price predictor by diving into the world of 
    **Multiple Linear Regression \(MLR\)** and **model selection**. We'll leverage **cross-validation** techniques and metrics like 
    **Akaike Information Criterion** $$(AIC),$$ **Bayesian Information Criterion**  $$(BIC)$$, **Predicted Residual Error Sum of Squares** $$(PRESS),$$ and **Adjusted Coefficient of Determination** $$(R^2_{Adj})$$ to find the 
    best model for our dataset of car prices, features, and other key attributes.

    In this tutorial, we'll explore the art of balancing model accuracy with simplicity. Cross-validation lets us validate 
    our model's performance on different data slices, providing a clearer view of how it might perform with new data. 
    Alongside, metrics like AIC and BIC help guide us toward models that not only fit well but also avoid overcomplicating things.

    Finally, we'll introduce tools to interpret feature importance through **t-statistics** and **SHAP values**, turning the 
    model's inner workings into actionable insights.

    Whether you're a data science enthusiast, a car aficionado, or just curious about machine learning, this blog will equip 
    you with hands-on tools and techniques to select the best regression model for predicting used car prices. 

    Buckle up as we rev up our data engines and hit the road to better predictions!
    """)
    

    st.markdown("""
    ### Brief Introduction to Multiple Linear Regression

    **Multiple Linear Regression (MLR)** is a statistical technique used to model the relationship between one dependent variable and multiple independent variables. Unlike simple linear regression, which examines just one predictor, MLR allows us to account for multiple factors at once, making it especially valuable for complex datasets.

    In MLR, we fit a line to minimize the difference between observed values and predicted values. The model can be expressed with the formula:

    $
    y = \\beta_0 + \\beta_1X_1 + \\beta_2X_2 + ... + \\beta_{p-1}X_{p-1} + \\epsilon
    $

    where:
    - $ Y $ is the dependent variable,
    - $( X_1, X_2, ..., X_{p-1} )$ are the independent variables,
    - $( \\beta_0 )$ is the intercept,
    - $( \\beta_1, \\beta_2, ..., \\beta_{p-1} )$ are the coefficients (slopes) representing the impact of each predictor on $Y $,
    - $( \epsilon )$ is the error term, capturing variations not explained by the model.

    MLR is particularly useful in predicting outcomes based on several factors, identifying significant predictors, and understanding the relative importance of each factor.

    #### Handling Categorical Variables in Multiple Linear Regression

    In multiple linear regression, categorical variables represent categories or groups (e.g., color, brand, or location) rather than numeric values. Since regression models require numeric inputs, we transform categorical variables into numerical formats to include them in the model. This is typically done through **dummy variables**

    #### Dummy Variables
    To include a categorical variable with \( k \) categories in a regression, we create \( k-1 \) binary (0 or 1) variables, or "dummy" variables, which indicate the presence of each category. For example, if we have a variable for "Car Color" with three categories (Red, Blue, Green), we create two dummy variables:
    - `Red`: 1 if the car is red, 0 otherwise.
    - `Blue`: 1 if the car is blue, 0 otherwise.

    The third category (Green) is the reference/baseline category, so when both `Red` and `Blue` are 0, the car is green.

    #### Incorporating Categorical Variables
    In regression, these dummy or one-hot encoded variables allow us to model the effect of each category on the outcome variable. By examining their coefficients, we can assess the impact of each category on the target variable while comparing it to the reference category.

    Handling categorical variables helps create a more interpretable and comprehensive model, as it allows us to capture and analyze the impact of group-based features.
    """)

    # Extensive introduction and detailed explanation
    st.markdown("""
    ## Cross-Validation in Multiple Linear Regression: An In-depth Exploration
    
    Cross-validation stands as a cornerstone technique in the realm of statistical analysis and predictive modeling. This method evaluates the efficacy of a model by partitioning the original dataset into complementary subsets, where one subset is used to train the model and the other to test its performance. This validation approach is essential for verifying the generalizability and robustness of predictive models, ensuring they perform consistently across various sets of data.
    
    ### Why Embrace Cross-Validation?
    - **Enhanced Model Evaluation**:
      - Cross-validation provides a more comprehensive assessment of a model's performance compared to a simple train/test split. It ensures that the model's effectiveness is tested across all data points, mitigating the risk of anomalies in any single train-test partition.
    - **Bias-Variance Tradeoff**:
      - It addresses the critical challenge in model training: balancing complexity and accuracy. Cross-validation helps in minimizing overfitting, ensuring the model performs well on any unseen dataset, not just the data on which it was trained.
    - **Model Selection**:
      - Through its iterative process, cross-validation aids in selecting the model that best handles new, unseen data. It allows for comparing the performance of different models on the same dataset, facilitating informed decision-making about the optimal model choice.

    ### Types of Cross-Validation
    - **K-Fold Cross-Validation**:
      - This method involves dividing the data into ‘K' subsets or folds. Each subset serves as a test set once, while the remaining subsets are used as the training set. This cycle repeats ‘K' times, with each of the ‘K' subsets used exactly once as the test set. The average of the results from all ‘K' folds provides a comprehensive performance estimate.
    - **Stratified K-Fold Cross-Validation**:
      - An enhancement of the K-fold method, stratified cross-validation ensures that each fold is a good representative of the whole by having approximately the same percentage of samples of each target class as the complete set.
    - **Leave-One-Out (LOO) Cross-Validation**:
      - In LOO, each observation is used as a separate test set, while the rest serve as the training set. This method is particularly beneficial when the dataset is small, as it maximizes the training data. However, it can be computationally expensive.

 
    ### Car Price Prediction User Case
      - In the context of predictive modeling and statistical analysis, cross-validation serves as a cornerstone technique that assesses a model's performance by partitioning the original dataset into complementary subsets—where one subset is used to train the model, and the other is used to test its accuracy. This approach is particularly important for ensuring the generalizability and robustness of predictive models, verifying that they perform consistently across different sets of data.

      - For our "Car Price Prediction Challenge," we utilize a dataset comprising 19,237 rows and 18 columns, with features ranging from basic car attributes like manufacturer, model, and production year, to specific details such as engine volume, mileage, and the number of airbags. The target variable, "Price," represents the car's selling price, which we aim to predict using various regression models.

      - The cross-validation process in this project is essential for determining how well our model can predict car prices on unseen data. By systematically splitting the dataset into training and testing subsets, we can evaluate different machine learning algorithms and fine-tune hyperparameters to enhance the model's predictive power. This ensures that the model does not merely memorize the training data but instead learns patterns that generalize well to new cars with varying features.
      """)
    st.markdown("""
        ### Visualizing K-Fold Cross Validation
        - In each fold, the model is trained on **K-1** subsets (blue) and tested on the remaining subset (red).
        - The process repeats for each fold, providing an average performance metric, giving a more reliable evaluation than a single train-test split.
        - Adjust the slider below to see how different values of **K** affect the model's performance across folds.
    """)
    
        # Data generation and plotting
    x, y = generate_data()
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='blue', alpha=0.5)
    ax.set_title("Generated Dataset")
    ax.set_xlabel("Predictor")
    ax.set_ylabel("Response")
    ax.grid(True)

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
    ## Steps of Cross-Validation: A Step-by-Step Guide for Car Price Prediction

    #### Step 1: Prepare Your Data
    For our Car Price Prediction Challenge, we start with a dataset of 19,237 rows and 18 columns, featuring attributes such as car manufacturer, model, production year, engine volume, mileage, number of airbags, and more. The target variable is the car's "Price," which we aim to predict using regression techniques.

    - **Data Cleaning:** Address missing values, handle duplicates, and ensure consistency across all attributes. For instance, columns like "Mileage" may have outliers that need to be addressed, while "Levy" or "Engine volume" might have missing values that require imputation.

    - **Feature Selection and Engineering:** Perform exploratory data analysis to identify important features. You may create new features, such as age of the car (calculated from the production year) or adjust categorical variables like "Fuel type" using one-hot encoding, to enhance the model's predictive power.

    #### Step 2: Split the Data into K Folds
    To ensure the robustness of our car price prediction model, we employ K-fold cross-validation. This technique systematically splits the dataset into K equal subsets (folds), where each fold serves as a testing set once while the remaining K-1 folds are used for training.

    - **Divide Equally:** The dataset is divided into five equal folds (`k=5`), meaning each fold contains approximately 3,847 rows of car data.

    - **Random Shuffling:** Shuffle the dataset before splitting to avoid any order-related biases, as demonstrated in the code using `np.random.permutation`.

    #### Step 3: Execute the Cross-Validation
    For each fold, the model training and testing proceed as follows:

    - **Training Set Creation:** Use 4 of the 5 folds as the training set, which consists of about 15,390 rows.

    - **Testing Set Selection:** The remaining fold, with approximately 3,847 rows, is used as the testing set to evaluate the model's predictive accuracy.

    - **Model Training:** Train an Ordinary Least Squares (OLS) regression model using the training set. In the code, this is done with `sm.OLS(y_train, X_train).fit()`, where `X_train` and `y_train` represent the training features and target variable (Price).

    - **Model Testing:** Test the trained model on the validation set (testing fold), calculating the mean squared error (MSE) to assess performance. MSE measures the average squared difference between the predicted car prices and the actual prices, providing insight into the model's accuracy.

    #### Step 4: Evaluate and Aggregate the Results
    After iterating through all 5 folds, the model's performance metrics are averaged:

    - **Performance Metrics:** Record the MSE for each fold and compute the average MSE across all folds to measure the model's predictive accuracy on unseen data.

    - **Additional Metrics (Optional):** Other evaluation metrics such as $$R^2, R^2_{Adj}, AIC, BIC,$$ and $$PRESS$$ for a more nuanced understanding of model performance.

    #### Step 5: Model Selection and Adjustment
    Based on the cross-validation results, make adjustments:

    - **Compare Models:** If multiple models are tested, compare the aggregated MSE or other metrics to choose the best-performing model. For example, you might test models with different predictors. 

    #### Step 6: Final Model Training and Validation
    - **Train Final Model:** Once the best model and parameters are identified, train the final model on the entire dataset to maximize learning, ensuring all available data contribute to the final model.

    - **External Validation (Optional):** If additional unseen data is available, validate the final model on this external dataset to confirm its generalizability and ensure it accurately predicts car prices for new data points.
    """)
    
    st.markdown("""
    ## Criteria for Selecting the Best Model in MLR
    In multiple linear regression (MLR), selecting the best model involves balancing fit quality and model simplicity.
    Here are four commonly used criteria for evaluating models, with their formulas:

    - **Akaike Information Criterion (AIC)**:
      - AIC evaluates models by considering the likelihood of the model and the number of predictors, penalizing model complexity. 
      -  $ \\text{AIC} = 2k - 2\ln(L) $
          - where $ k $ is the number of parameters (including the intercept) and $ L $ is the maximum likelihood of the model.
      - Lower AIC values indicate a better balance between fit and complexity.

    - **Bayesian Information Criterion (BIC)**:
      - Similar to AIC, BIC applies a stronger penalty for models with more parameters, especially useful with larger sample sizes. 
      - $ \\text{BIC} = k \ln(n) - 2\ln(L) $  
          - where $ n $ is the number of observations and $ k $ is the number of parameters.
      - Lower BIC values indicate a preferred model, particularly for larger datasets.

    - **Predicted Residual Sum of Squares (PRESS)**:
      - PRESS evaluates predictive accuracy by measuring how well the model predicts unseen data points. 
        $ \text{PRESS} = \sum_{i=1}^{n} (y_i - \hat{y}_{-i})^2 $  
      where $ y_i $ is the observed value for each point, and $ \hat{y}_{-i} $ is the predicted value for the $ i $-th observation, excluding it from the model fitting.
      - Lower PRESS values suggest better predictive power.

    - **Adjusted R-Squared ($ R^2_{Adj} $)**:
      - Adjusted $ R^2 $ accounts for the number of predictors, adjusting the $ R^2 $ value to avoid overfitting. 
      - $ R^2_{Adj} = 1 - \\left(1 - R^2\\right) \\frac{n - 1}{n - k - 1} $  
          - where $ R^2 $ is the coefficient of determination, $ n $ is the number of observations, and $ k $ is the number of predictors.
      - Higher adjusted $ R^2 $ values indicate a model that explains more variance without unnecessary complexity.

    By using these criteria together, we can select a model that balances predictive accuracy, interpretability, and simplicity.
    """)



if __name__ == "__main__":
    main()