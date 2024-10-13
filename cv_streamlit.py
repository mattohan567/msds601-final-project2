import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import statsmodels.api as sm
import plotly.express as px
import statsmodels.formula.api as smf
import ast
import re




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

criterias = ['neg_mean_squared_error']
def perform_cross_validation(x, y, k):
    """
    Performs K-fold cross-validation on a simple linear regression model.
    """
    model = LinearRegression()
    scores = {}
    for criteria in criterias:
        scores[criteria] = cross_val_score(model, x, y, cv=k, scoring=criteria)
    return scores

results_df = pd.read_csv('data/all_results_df.csv')

df = pd.read_csv('data/cleaned_car_price_prediction_door_fix.csv')
df['HasTurbo'] = df['HasTurbo'].astype(int)
df.columns = [re.sub(' ', '_', col) for col in df.columns]
df.columns = [re.sub('\.', '', col) for col in df.columns]
x_cols = df.columns[1:]
df = df.sample(n=2000, random_state=42)

best_models = {
    'MSE': results_df.sort_values('MSE').iloc[0],
    '5-Fold CV MSE': results_df.sort_values('5-Fold_CV MSE').iloc[0],
    '10-Fold CV MSE': results_df.sort_values('10-Fold_CV MSE').iloc[0],
    'AIC': results_df.sort_values('AIC').iloc[0],
    'BIC': results_df.sort_values('BIC').iloc[0],
    'Adjusted R^2': results_df.sort_values('Adjusted R^2', ascending=False).iloc[0],
    'PRESS': results_df.sort_values('PRESS').iloc[0]
}

# Function to create formula from predictors
def create_formula(predictors_str):
    # Convert string representation of list to actual list
    predictors = ast.literal_eval(predictors_str)
    
    formula = "Price ~ " + " + ".join([f"C({p})" if df[p].dtype == 'object' else p for p in predictors])
    return formula

# Create formulas for each best model
best_formulas = {criterion: create_formula(model['Predictors']) for criterion, model in best_models.items()}

fitted_models = {}

for criterion, formula in best_formulas.items():
    model = smf.ols(formula, data=df).fit()
    fitted_models[criterion] = model


def plot_variable_importance_interactive(model, title, n=5):
    """
    Function to create an interactive Plotly bar chart showing the variable importance
    with coefficients included in the hover information.
    """
    importance = abs(model.tvalues)[1:]  # Exclude intercept
    importance = importance.sort_values(ascending=False)  # Sort descending
    
    if n is not None and n > 0:
        importance = importance.head(n)  # Get the top n highest values
    
    # Create a DataFrame to include both t-statistics and coefficients
    data = pd.DataFrame({
        'Variable': importance.index,
        '|t-statistic|': importance.values,
        'Coefficient': model.params[importance.index]
    })

    # Create the bar chart with Plotly
    fig = px.bar(
        data, 
        x='|t-statistic|', 
        y='Variable', 
        orientation='h', 
        hover_data={'Coefficient': True, '|t-statistic|': True},
        title=f'Top {n} Variable Importance based on t-statistic: {title}'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})  # Ensure sorted plot
    
    # Display the chart in Streamlit
    st.plotly_chart(fig)

# Interactive section for Variable Importance Charts
def interactive_variable_importance_section():
    st.header("Interactive Variable Importance Charts")
    st.write("Hover over each bar to see details about its t-statistic and coefficient.")

    # Loop over the models and show interactive charts
    for criterion, model in fitted_models.items():
        plot_variable_importance_interactive(model, criterion, n=5)  # Show top 5 variables by default




def main():
    st.title("""Predicting Car Prices 🚗: A Journey Through Cross-Validation and Model Selection in Regression""")
    st.markdown("*By Tim Geum, Georgia von Minden, Matt Ohanian, and Iris Yu*")
    
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

    Finally, we'll introduce tools to interpret feature importance through **t-statistics**, turning the 
    model's inner workings into actionable insights.

    Whether you're a data science enthusiast, a car aficionado, or just curious about machine learning, this blog will equip 
    you with hands-on tools and techniques to select the best regression model for predicting used car prices. 

    Buckle up as we rev up our data engines and hit the road to better predictions!
    
    (Topic Extension: Cross Validation in MLR, Predictor Importance)
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
    
    st.markdown("""
    ## Dataset: Car Price Prediction
    This analysis uses data from the [Car Price Prediction Challenge on Kaggle](https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge), which contains features that help predict car prices. 

    ### Dataset Description
    The dataset includes various car attributes such as:
    - **Manufactorer**: The brand of each car.
    - **Year**: The year of manufacture.
    - **Category**: The type of car (i.e. sedan)
    - **Leather Interior**: If the car has a leather interior
    - **Has Turbo**: If the car has a turbo engine
    - **Engine Size**: Engine specifications that can influence price.
    - **Fuel Type**: The type of fuel used, such as petrol, diesel, etc.
    - **Cylinders**: The number of cylinders in the engine
    - **Mileage**: The total distance a car has traveled, affecting resale value.
    - **Transmission Type**: Automatic or manual transmission, which can influence price preferences.
    - **Doors**: The number of doors, which may affect the car's appeal to certain buyers.
    - **Gear Box Type**: The transmission type of the car
    - **Drive Wheels**: Which wheels drive (i.e. Front)
    - **Wheel**: The side of the car the driving wheel is on
    - **Color**: The color of the car
    - **Airbags**: The number of airbags in a car
    - **Price**: The target variable, representing the car's price.
    

    ### Objective
    Our goal is to use multiple linear regression (MLR) to predict car prices based on these attributes, selecting the best model through various criteria like AIC, BIC, PRESS, and Adjusted R².

    By using this dataset, we aim to identify the most impactful features on car price and provide a predictive model that balances interpretability and predictive power.
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
        - We will show this if we trained a model using all of the predictors provided on a subset of 5000 observations
        - Adjust the slider below to see how different values of **K** affect the model's performance across folds.
    """)
    
        # Data generation and plotting
    # x, y = generate_data()
    df = pd.read_csv('data/cleaned_car_price_prediction_door_fix.csv')
    subset = df.sample(n=5000, random_state=42)
    y = subset['Price']
    x = subset.iloc[:, 1:]
    cat_cols = [col for col in x.columns if subset.dtypes[col] == 'object']
    x = pd.get_dummies(x, cat_cols, drop_first=True)
    x = sm.add_constant(x, has_constant='add')

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
        - MSE (Mean Squared Error) is the average of the squared residuals/errors for each observation in the validation/test set

    - **Additional Metrics (Optional):** Other evaluation metrics such as $$R^2, R^2_{Adj}, AIC, BIC,$$ and $$PRESS$$ for a more nuanced understanding of model performance.

    #### Step 5: Model Selection and Adjustment
    Based on the cross-validation results, make adjustments:

    - **Compare Models:** If multiple models are tested, compare the aggregated MSE or other metrics to choose the best-performing model. For example, you might test models with different predictors. 

    #### Step 6: Final Model Training and Validation
    - **Train Final Model:** Once the best model and parameters are identified, train the final model on the entire dataset to maximize learning, ensuring all available data contribute to the final model.

    - **External Validation (Optional):** If additional unseen data is available, validate the final model on this external dataset to confirm its generalizability and ensure it accurately predicts car prices for new data points.
    """)
    
    st.markdown("""
    ## Other Criteria for Selecting the Best Model in MLR
    In multiple linear regression (MLR), selecting the best model involves balancing fit quality and model simplicity.
    Here are four commonly used criteria for evaluating models, with their formulas:

    - **Akaike Information Criterion (AIC)**:
      - AIC evaluates models by considering the likelihood of the model and the number of predictors, penalizing model complexity. 
      -  $ \\text{AIC} = 2p - 2\ln(L) $
          - where $ p $ is the number of parameters (including the intercept) and $ L $ is the maximum likelihood of the model.
      - Lower AIC values indicate a better balance between fit and complexity.\n

    - **Bayesian Information Criterion (BIC)**:
      - Similar to AIC, BIC applies a stronger penalty for models with more parameters, especially useful with larger sample sizes. 
      - $ \\text{BIC} = p \ln(n) - 2\ln(L) $  
          - where $ n $ is the number of observations and $ p $ is the number of parameters.
      - Lower BIC values indicate a preferred model, particularly for larger datasets.\n

    - **Predicted Residual Sum of Squares (PRESS)**:
      - PRESS evaluates predictive accuracy by measuring how well the model predicts unseen data points. 
        $ \\text{PRESS} = \sum_{i=1}^{n} (y_i - \hat{y}_{-i})^2 $  
      where $ y_i $ is the observed value for each point, and $ \hat{y}_{-i} $ is the predicted value for the $ i $-th observation, excluding it from the model fitting.
      - Lower PRESS values suggest better predictive power.\n

    - **Adjusted R-Squared ($ R^2_{Adj} $)**:
      - Adjusted $ R^2 $ accounts for the number of predictors, adjusting the $ R^2 $ value to avoid overfitting. 
      - $ R^2_{Adj} = 1 - \\left(1 - R^2\\right) \\frac{n - 1}{n - p - 1} $  
          - where $ R^2 $ is the coefficient of determination, $ n $ is the number of observations, and $ p $ is the number of predictors.
      - Higher adjusted $ R^2 $ values indicate a model that explains more variance without unnecessary complexity.

    By using these criteria together, we can select a model that balances predictive accuracy, interpretability, and simplicity.
    """)

    st.markdown("""
    ## Model Selection Methodology

    To build a high-performing used car price predictor, we developed a systematic approach to model selection that balances predictive accuracy, interpretability, and computational efficiency. Here’s an overview of our methodology:

    1. **Data Subsetting**:
      - The original dataset contained 19,000 observations. To optimize for time and computational resources, we randomly sampled 2,000 observations. This subset maintains data representativeness while reducing processing demands, enabling us to train and evaluate multiple models effectively.

    2. **Exhaustive Subset Evaluation**:
      - With 15 predictors, there are \(2^{15} - 1 = 32,767\) possible combinations. We trained a model for each combination, evaluating each subset to capture a thorough view of how different predictors impact model performance.
      - This approach allowed us to comprehensively assess every possible subset, identifying optimal models that balance accuracy with interpretability.

    3. **Evaluation Metrics**:
      We evaluated each model subset using the following five metrics:
      - **Adjusted \( R^2 \)**
      - **Akaike Information Criterion (AIC)**
      - **Bayesian Information Criterion (BIC)**
      - **Predicted Residual Sum of Squares (PRESS)**
      - **5-Fold and 10-Fold Cross-Validation MSE**

    4. **Selecting the Best Model**:
      - For each metric, we identified the best-performing model. Most metrics, including Adjusted \( R^2 \), AIC, PRESS, and both cross-validation MSEs, favored models with approximately **69 predictors**, suggesting that this number provides a solid balance of accuracy, interpretability, and predictive power.
      - **Consistency Between 5- and 10-Fold Cross-Validation**: Both cross-validation metrics identified the same optimal model, which is a strong indicator of its stability and generalizability to new data.
      - **BIC Divergence**: Unlike the other criteria, BIC preferred models with 12 or 13 predictors due to its stronger penalty on model complexity, which is in line with BIC’s goal of minimizing overfitting by favoring simpler models.

    5. **Interpretation and Trade-offs**:
      - The agreement across most metrics for models with 69 predictors is a strong indicator of robustness. These models provide a reliable balance between complexity and predictive power.
      - The BIC-favored models with 12-13 predictors offer a simpler alternative, useful in cases where model interpretability or complexity is a higher priority.

    By systematically evaluating every possible subset and comparing results across these metrics, we were able to select models that generalize well to unseen data and strike a balance between accuracy and simplicity, ensuring practical applicability for predictive analytics.
    """)
    
    st.markdown("""
    ## Understanding Feature Importance in Multiple Linear Regression

    When working with multiple linear regression (MLR), we encounter multiple features (independent variables) that predict a target (the dependent variable). However, not all features contribute equally—some have a strong influence, while others may have minimal or even negative effects on the model’s predictions. Understanding feature importance helps us refine our model, focusing on the most impactful variables to create a simpler and more effective prediction tool.

    Coefficients as a Measure of Feature Importance

    In MLR, the coefficients for each feature provide insight into its effect on the target variable. A larger absolute value for a coefficient indicates a stronger influence. For example, in predicting car prices, a large negative coefficient for “Mileage” would suggest that cars with higher mileage tend to have significantly lower prices. Conversely, there could be a positive coefficient for a specific categorical variable's state (eg: brand being a luxury namesake like Mercedes).

    The linear regression model is represented as:

    $$
    \hat{y} = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + \\dots + \\beta_n x_n
    $$

    Where:
    - $\hat{y}$ is the predicted target (e.g., car price),
    - $\\beta_0$ is the intercept (the value when all features are zero),
    - $\\beta_1, \\beta_2, \\dots, \\beta_n$ are the coefficients for the features.
    
    A large coefficient means the feature has a strong relationship with the target. The sign (positive or negative) tells us the direction of that relationship. But coefficients alone don’t give the full picture, especially when features are on different scales. That’s where t-statistics come in.

    T-Statistics: Gauging Feature Impact

    The t-statistic helps us determine whether a feature’s coefficient is meaningful or just noise. It tests whether the feature significantly contributes to the model.

    The t-statistic is calculated as:

    $$ t = \\frac{\\hat{\\beta}}{SE(\\hat{\\beta})} $$

    Where:

    $\hat{\\beta}$ is the feature’s coefficient, and 
    $SE(\hat{\\beta})$ is the standard error of the coefficient.

    A larger t-statistic (in absolute terms) means we can be more confident that the feature is genuinely affecting the target variable.

    P-Values: Determining Statistical Significance

    The p-value, derived from the t-statistic, tells us if the feature’s impact is statistically significant:

      -	A small p-value (typically < 0.05) indicates the feature is likely important. \n
      -	A large p-value suggests the feature may not be contributing meaningfully and could be considered for removal.

    Balancing both coefficients and p-values helps us decide which features to retain. A feature with a high p-value might seem important based on its coefficient but could be unreliable. Conversely, a feature with a small coefficient and low p-value may still have a subtle but consistent impact.

    Visualizing Feature Importance

    A simple way to compare feature importance is by visualizing the absolute values of t-statistics or coefficients in a bar plot. This helps identify which features have the most influence at a glance.

    Aligning with Intuition

    Our model confirms that features like Manufacturer, Leather, and other perks like larger engines significantly affect car resale prices, which aligns with common expectations:

      -	Cars with more amenities (like airbags or turbo engines) are generally more valuable.
      -	Manufacturer reputation plays a crucial role—certain brands command higher resale values.
      -	Lower mileage and newer production years correlate with higher prices due to reduced wear and tear.

    By verifying these factors with our model, we can show how data-driven insights confirm well-known assumptions about car values.
    """)
    interactive_variable_importance_section()

    st.markdown("""
    
    ## Conclusion

    As a result of our exercise, we answered our original question: "How much should this car cost?"

     - **First**, we used cool model selection criteria such as k-fold cross validation (5 and 10 folds in our case) as well as other criteria such as AIC, BIC, and PRESS we learned in class. Finally, we trained the model on our dataset and extracted the best models using each criteria. This ultimately gave us a set of best models each with a set of features that predict the price of used cars. 

     - **Second**, we used the feature importance tool 'variable importance plot' which determines how important each variable is in a model and how it relates to the predicted price of a used car. Using this plot, we saw some "repeat customers" in terms of top predictors that have high impact in determining a used car price across all models.
        - Production year: The age of the car, unsurprisingly, is the top most important predictor for the price of a new car for all of our best models.
        - Gearbox type: When the type was tiptronic, the gearbox seems to be the second most important predictor for a new car for most of our best models.
        - Airbags: While airbags are crucial for safety, it had a higher impact in determining car price than expected.
        - Manufacturer: We definitely do assume manufacturer (brand name) to be one of the strongest predictors for used car prices, but what was surprising is that impact was highest when the brand is GAZ which are known to make cheap cars.
        - Engine turbo: A strong predictor in car price, especially for prospective buyers who care about car perfromance.
    
    One of the interesting things in this exercise is that we do not see mileage to be the top 5 predictors of car prices, at least in our models. It's possible that, because mileage has a wide scatter (0 to 100,000km+), its effect was relatively unimportant. There are other things that could have made our exercise more complex and interesting such as the degree of wear-and-tear, parts replaced, remodeling, as well as supply/demand. 

    But for the purpose of our exercise, we have derived some strong models using some fundamental linear regression principles and learned the top features that play the most roles in determining car prices! It's very interesting to see that some features are as important as expected but some played more important roles than expected while other features were less important than expected. Using this example, I hope you learned more about choosing the best linear regression models as well as determining which features play the most important part in determining the response of that model!

    """)

    st.markdown("*Happy Hunting!*")

if __name__ == "__main__":
    main()