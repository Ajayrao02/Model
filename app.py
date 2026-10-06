import pandas as pd
from sklearn.model_selection import train_test_split
#70-30, 80-20, 90-10
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import streamlit as st


st.set_page_config(page_title="Mark Prediction", page_icon="📃")
st.title("Marks Prediciton System")
st.subheader("Hello")
st.write("Predicted marks are on study time")
data = {
    "Hrs": [1,2,3,4,5,6,7,8,9,10],
    "Marks": [10, 20, 40, 30, 30, 33, 50, 80, 90, 100]
}

df = pd.DataFrame(data)
with st.expander("View Data set"):
    st.dataframe(df)

st.subheader("Data set Info")
col1, col2, col3 = st.columns(3)
with col1:
    st.info(len(df))
with col2:
    st.success(df.duplicated().sum())
with col3:
    st.error(df.isnull().sum())

x = df[["Hrs"]]
y = df["Marks"]

x_train , x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
# print(x_train)
model = LinearRegression()
model.fit(x_train, y_train)

pred = model.predict(x_test)
score = r2_score(pred, y_test)

# hrs = st.number_input("Enter your study Hours:", min_value=5, max_value=24, value=10)
hrs = st.number_input("Enter your study Hours:", min_value=5, max_value=24, value=10, step=3)
hrs = st.slider("Enter your study hours", 0, 23 ,10)
if st.button("Predict", use_container_width=True):
    prediction = model.predict([[hrs]])
    st.success(f"Your predicted marks is : {prediction[0]:.2f}")
    st.info(f"Modal accuracy is : {round(score, 4)}")


tab1, tab2 =st.tabs(["Plot Graph", "Scatter graph"])
with tab1:
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(df["Hrs"], df["Marks"])
    # ax.scatter(df["Marks"], model.predict(x))
    st.pyplot(fig)
with tab2:
    fig, ax = plt.subplots(figsize=(6,4))
    # ax.plot(df["Hrs"], df["Marks"])
    ax.scatter(df["Marks"], model.predict(x))
    st.pyplot(fig)


st.caption("Developed by SpideyWeb")

    # # plt.plot(x, y, color="red")
    # plt.plot(df["Hrs"], df["Marks"])
    # # plt.scatter(model.predict(x), y, color="purple")
    # plt.scatter(df["Marks"], model.predict(x))
    # plt.scatter(hrs, prediction)
    # plt.show()


