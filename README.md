**Live Site:** [Live webpage]()

**Link to Repository:** [Repository](https://github.com/michmattera/heart-disease-analysis-and-prediction)

## Table of Content

- [Table of Content](#table-of-content)
- [Introduction](#introduction)
- [IDE Reminders](#ide-reminders)
- [Dataset Content](#dataset-content)
- [CRISP-DM](#crisp-dm)
- [Business Requirements](#business-requirements)
- [Hypothesis and how to validate?](#hypothesis-and-how-to-validate)
- [The rationale to map the business requirements to the Data Visualizations and ML tasks](#the-rationale-to-map-the-business-requirements-to-the-data-visualizations-and-ml-tasks)
- [ML Business Case](#ml-business-case)
- [Dashboard Design](#dashboard-design)
  - [Page 1: Quick project summary](#page-1-quick-project-summary)
  - [Page 2: Review Analysis Summary](#page-2-review-analysis-summary)
  - [Page 3: Review Prediction](#page-3-review-prediction)
  - [Page 4: Hypothesis ad validation](#page-4-hypothesis-ad-validation)
  - [Page 5: Information on ML used](#page-5-information-on-ml-used)
- [Unfixed Bugs](#unfixed-bugs)
- [Deployment](#deployment)
  - [Heroku](#heroku)
- [Main Data Analysis and Machine Learning Libraries](#main-data-analysis-and-machine-learning-libraries)
- [Credits](#credits)
  - [Content](#content)
  - [Media](#media)
- [Acknowledgements (optional)](#acknowledgements-optional)

**Developed by: Michelle Mattera**

## Introduction

This machine learning project was developed for the fifth portfolio project during the Code Insititute's Diploma in Full Stack Development. It covers the Predictive Analytics specialization.

The machine and data analysis was created from the [Heart disease dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset). The purpose of this machine learning project was to allow the user to predict heart disease based on a combination of features. In addiction it shows to the user how these features correlate with the final prediction.

## IDE Reminders

To log into the Heroku toolbelt CLI:

1. Log in to your Heroku account and go to _Account Settings_ in the menu under your avatar.
2. Scroll down to the _API Key_ and click _Reveal_
3. Copy the key
4. In the IDE terminal, run `heroku_config`
5. Paste in your API key when asked

You can now use the `heroku` CLI program - try running `heroku apps` to confirm it works. This API key is unique and private to you so do not share it. If you accidentally make it public then you can create a new one with _Regenerate API Key_.

## Dataset Content

The dataset is sourse from [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

## CRISP-DM

This project was developed using the Cross Industry Standard Process for Data Mining. Developer choose to divide the steps following an agile method and dividing the steps in Epics.

1. Epic 1: Business Understanding - This incorporates understanding the clients business case, usually through a conversation with the client where it is establish the business case and decide together the acceptance criteria.
2. Epic 2: Data Understanding - The data needed to achieve the business requirements must be identified and understood. After data collection the data needs to be find, cleaned and checked if with the data is possible to solve or be used for the business requirements discussed above.
3. Epic 3: Data Preparation - .
4. Epic 4: Modelling - .
5. Epic 5: Evaluation - .
6. Epic 6: Deployment - Develop the streamlit app that will satisfy the business requirements determined in collaboration with the client and deploy the app online. The app is deployed in Heroku and the process is described in the Deployment section below.

## Business Requirements

The client is a Global Health Organization that would like to investigate and predict if a user would likely suffer from a heart disease or not. The client is trying to understand as well which features are more related to the prediction and why. The organization is trying to understand the pattern to be able to advert subject at risk , and take as many precautions as possible with their patients.
The business requirements were discussed with the client .

1. The client is interested in understanding the patterns from the heart disease database so that the client can learn the most relevant variables correlated to a positive heart desease prediction.
2. The client is interested in determining if a patient would have heart disease or not.

## Hypothesis and how to validate?

- List here your project hypothesis(es) and how you envision validating it (them)

## The rationale to map the business requirements to the Data Visualizations and ML tasks

- List your business requirements and a rationale to map them to the Data Visualizations and ML tasks

## ML Business Case

- In the previous bullet, you potentially visualized an ML task to answer a business requirement. You should frame the business case using the method we covered in the course

## Dashboard Design

### Page 1: Quick project summary

- Quick project summary
  - Small heart picture
  - Project Terms & Jargon
  - Describe Project Dataset
  - State Business Requirements

### Page 2: Review Analysis Summary

- After data analysis, we agreed with stakeholders that the page will:
  - State business requirement 1
  - Data inspection on the first 10 rows of the dataset used
  - Display the most correlated variables to the target and the conclusions
  - Checkbox: Showing first correlation study
  - Checkbox: Showing second correlation study

### Page 3: Review Prediction

- The third page will display the second business requirement
  - State business requirement 2
  - Inputs with most important features
  - Each input can be modified to have different result
  - Button where client can predict the heart deseaseof a patient based on the inputs
  - After cliking the predict button , the ml pipeline will use that set of inputs for prediction. The client will have display the result, if is going to have a good review or not and the percentual.

### Page 4: Hypothesis ad validation

- The fourth page will have display all the hypothesis and validation for the entire project
  - State hypothesis
  - State validation

### Page 5: Information on ML used

- The fifth page will display the following:
  - Which model was used
  - Description of the ML pipeline
  - Demonstration of features importance
  - Performance overview

## Unfixed Bugs

- You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable to consider, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

## Deployment

### Heroku

- The App live link is: <https://YOUR_APP_NAME.herokuapp.com/>
- Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

- Here you should list the libraries you used in the project and provide an example(s) of how you used these libraries.

## Credits

- In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism.
- You can break the credits section up into Content and Media, depending on what you have included in your project.

### Content

- The text for the Home page was taken from Wikipedia Article A
- Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

- The photos used on the home and sign-up page are from This Open-Source site
- The images used for the gallery page were taken from this other open-source site

## Acknowledgements (optional)

- Thank the people that provided support through this project.
