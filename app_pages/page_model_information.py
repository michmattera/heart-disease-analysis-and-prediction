import streamlit as st
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from feature_engine.transformation import YeoJohnsonTransformer
from src.data_management import load_heart_disease_data, load_pkl_file
from src.machine_learning.evaluate_performance import clf_performance


def model_information_body():

    # Load all files
    version = "v3"
    pipeline = load_pkl_file(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/xgbclassifier_pipeline.pkl")
    best_features = pd.read_csv(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/X_train.csv").columns.to_list()
    classification_report_image = plt.imread(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/classification_report_3.png")
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/X_train.csv")
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/X_test.csv")
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/y_train.csv").values
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_heart_disease/{version}/y_test.csv").values

    st.write("## Model Information")

    # Introduction section
    st.info(
        f"Welcome to the Machine Learning Model Dashboard."
        f" This dashboard provides an overview of our model and its performance. \n"
        f"\n"
        f"Developer tried different"
        f" approaches and features to find best model, with higher performance. \n"
        f"\n"
        f"In this page you will find : \n"
        f"\n"
        f"* Model information \n"
        f"* Model performance \n"
        f"* Confusion matrix report \n"
        f"* Confusion matrix heatmap \n"
    )

    hyperparameters = {
        'learning_rate': 0.01,
        'max_depth': 3,
        'n_estimators': 100
    }

    # Model Information section
    st.header("Model Information")
    st.markdown("- The type of model : xgbclassifier")
    st.markdown("- Features used in the model :")
    st.write(best_features)
    st.markdown("- Training data source : ")
    st.write(X_train)
    st.markdown("- Model hyperparameters: ")
    st.write(hyperparameters)
    st.markdown("- Model transformation :")
    if st.checkbox("Model transformation QQ plot"):
        st.write(
            f"Here you can see the difference of the variable before and after transformation, needed to reach better performance."
        )
        create_qq_plot()

    # show pipeline steps
    st.write("* **This is the final ML pipeline used to predict a heart disease**")
    st.write(pipeline)

    # Model Performance section
    st.header("Model Performance")
    st.write("Final model performance meet all the criteria discussed with client."
             )

    st.info(
        f" **Pass performance criteria discussed with client:** \n"
        f"\n"
        f"* Precision of 0.85 \n"
        f"* Recall of 0.85 \n"
        f"* Accuracy of minimum 0.80 \n"
    )

    st.success(
        f"**Train set :** \n"
        f"* Precision of no disease of 0.87 \n"
        f"* Recall of  Disease of 0.91 \n"
        f"* Accuracy of 0.85 \n"
        f"\n"
        f"All of them passing the criterias set in the business requirement."
    )

    st.success(
        f"**Test set :** \n"
        f"* Precision of no disease of 0.90 \n"
        f"* Recall of  Disease of 0.93 \n"
        f"* Accuracy of 0.85 \n"
        f"\n"
        f"All of them exeeding the criterias set in the business requirement,"
        f" with a higher media than the train set reaching at least 0.90"
        f" in precision and recall."
    )

    if st.checkbox("Confusion matrix for train and test set"):
        st.write("### Pipeline Performance")
        clf_performance(x_train=X_train, y_train=y_train,
                        x_test=X_test, y_test=y_test,
                        pipeline=pipeline,
                        label_map=["No Disease", "Disease"])

    if st.checkbox("Classification Report Heatmap"):
        st.write(
            f"Here you can see the classification report for the test set performance."
        )
        st.image(classification_report_image)


# Code copy from FeatureEngineering notebook
def create_qq_plot():
    # Load data (replace with your data loading logic)
    df = load_heart_disease_data()

    variables_to_transform = ['cp', 'chol',
                              'thalach', 'exang', 'oldpeak', 'ca']
    # Create a transformer for Yeo-Johnson transformation
    transformer = YeoJohnsonTransformer(variables=variables_to_transform)
    transformed_data = transformer.fit_transform(df)

    # Streamlit app
    st.title("Q-Q Plot Visualization")

    # Display Q-Q plots for original and transformed data
    for variable in variables_to_transform:
        original_data = df[variable]
        transformed_variable = transformed_data[variable]

        # Plot Q-Q plot for original data
        plt.figure(figsize=(6, 3))
        stats.probplot(original_data, dist='norm', plot=plt)
        plt.title(f'Q-Q Plot for Original {variable}')
        st.pyplot(plt)

        # Plot Q-Q plot for transformed data
        plt.figure(figsize=(6, 3))
        stats.probplot(transformed_variable, dist='norm', plot=plt)
        plt.title(f'Q-Q Plot for Transformed {variable}')
        st.pyplot(plt)
