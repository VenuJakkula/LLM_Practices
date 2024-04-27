Streamlit web app for Recommendation using KMeans which includes model training and inferencing.
=====================================================================================================

Objective: This web app will take test query (Summary) as input produces recommendations as output

Installed libraries
--------------------------
pip install streamlit
pip install scikit-learn
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install faiss-cpu==1.7.4
pip install huggingface-hub==0.16.4
pip install langchain==0.0.235
pip install sentence-transformers==2.2.2
pip install tiktoken==0.5.1
pip install tokenizers==0.13.3
pip install transformers==4.33.2

Steps to create App:
- Install streamlit
- Create app.py and import streamlit
- design the layout of html components using streamlit
- App contains single page that takes an input and produces output in table format

steps to deploy:
- Streamlit can be deployed on various cloud platforms including streamlit cloud using github
- As of now we are serving this app on localhost

Steps to run app:
- Goto the path where app.py is located
- Run the command below to start the application.
	python -m streamlit run App.py

App Functioning:
- After app is loaded, it will show input box. 
- search button will be enabled after latest model is loaded successfully
- A toast message will be displayed after the model loading
- A spinner will be displayed untill model is loaded.
- once user enter a test query and click submit button
	- ground truth data (Bira links) will be shown if there is any data available.
	- recommendations table will be shown below.
- After every refresh a latest model will be loaded