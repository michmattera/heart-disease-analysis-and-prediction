import streamlit as st


def predict_heart_disease(X_live, best_features, pipeline):

    # subset feature variables
    features_live_vars = X_live.filter(best_features)

    # Make prediction
    heart_disease_prediction = pipeline.predict(
        features_live_vars)
    heart_disease_prediction_probability = pipeline.predict_proba(
        features_live_vars)

    # display results
    probability = heart_disease_prediction_probability[0,
                heart_disease_prediction][0]*100

    if heart_disease_prediction == 1:
        heart_disease_result = 'will'
    else:
        heart_disease_result = 'will not'

    statement = (
        f'### There is {probability.round(1)}% probability '
        f'that this patient **{heart_disease_result} suffer from heart disease**.')

    st.write(statement)

    return heart_disease_prediction
