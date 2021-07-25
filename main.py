# Libraries

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


#Set title of our Front web app

st.title('Rebel-Food Assignment')



#Set Slide bar to chose between EDA and Visualization
def main():
	activities=['Model building' , 'Ideation']
	option=st.sidebar.selectbox('Selection option:',activities)

	#EDA Tab
	if option =='Model building':
		st.subheader("Exploratory Data Analysis")

		#File Upload
		data=st.file_uploader("Upload you CSV or Excel File:",type=['csv','xlsx'])
		st.success("File Upload Success")

		if data is not None:
			file=pd.read_csv(data)
			st.dataframe(file.head(10))

			if st.checkbox("Show Data Columns"):
				st.write(file.columns)

			if st.checkbox("Show shape of Data"):
				st.write(file.shape)

			if st.checkbox("Show Missing values Shape"):
				st.write(file.isnull().isnull().sum())

			if st.checkbox("Count values of Target Variable"):
				st.write(file['sex'].value_counts())

			# Replacing All F and M with 0 and 1 respectively
			file.sex.replace({'F': 0, 'M': 1}, inplace=True)


			# Feature Extraction
			features = file['name']
			cv = CountVectorizer()
			X = cv.fit_transform(features)

			# Features
			#X
			# Labels
			y = file.sex

			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

			# Naive Bayes Classifier
			from sklearn.naive_bayes import MultinomialNB
			clf = MultinomialNB()
			clf.fit(X_train, y_train)
			clf.score(X_test, y_test)

			# Accuracy of our Model
			if st.checkbox("Accuracy  of Naive Bayes Classifier "):
				acc=("Accuracy of Model "+ str(clf.score(X_test, y_test) * 100)+ "%")
				st.write(acc)



				if st.checkbox("Want to make a single Name Gender prediction? "):
					user_input = st.text_input("Enter Name to predict the gender")
					sample = [user_input]
					vect = cv.transform(sample).toarray()
					if clf.predict(vect)==0:
						st.write("It's a Female")
					else:
						st.write("It's a male")





	if option =='Ideation':
		st.write("1. How will you build Such Model: ")
		st.write("A: This can be taken as a supervised Machine Learning problem based on the data available We Can use different Classification models sucha as Decision trees, Naive Bayes, Bagging and Boosting models like Random forest, XGBoost, Logistic regression etc")
		st.write("B: If target variable is not avaialbe and depending on Data we can take this as unsupervised clustering problem which can be solved using KMeans clustering")
		st.write("C: Deep learning technique such as ANN can also be used if vast data is availabe and Traditional ML models are under performing")

		st.write("2. What data points do you think would be relevant?")
		st.write("A. Delivery Agent's review rating for the particular customer in past")
		st.write("B. Age of the customer")
		st.write("C. Frequency of customer's past orders and its count")
		st.write("D. Rating of Restaurant where order is placed")
		st.write("E. Avg Time taken by Restaurant to cook the order")
		st.write("F. No of time user reached out to the customer care in the past ")
		st.write("G. Does user have premium type of subscription?")


		st.write("3. What features would you include in the model?")
		st.write("A. Ensemble modeling")
		st.write("B. Deployment with realtime learning and adaptation")




if __name__ == '__main__':
	main()
