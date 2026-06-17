import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.markdown("""
<style>
div.stButton > button {
    background-color: #00C853;
    color: white;
    border-radius: 12px;
    height: 40px;
    font-size: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

tab1 ,tab2 , tab3 ,tab4= st.tabs(
    ["Predictions" , "EDA" , "Model Metrics" , "Understanding Metrics"]
)
with st.sidebar:
    st.header("Model Information")

    st.write("Algorimth : Random Forest Regressor")
    st.write("Cross Val R² : 74%")
    st.write("MAPE : 13%")

with tab1:

    st.markdown(
    """
    #  Car Price Predictor

    Predict the market value of a car using Machine Learning.
    """
    )

    model = joblib.load("models/Random_forest_car_price_predictor.pkl")


    col1,col2= st.columns(2)

    with col1:

        brand = st.selectbox(
            "Brand",
            [
                'Acura', 'Aston Martin', 'Audi', 'Bentley', 
                'Bmw', 'Bugatti', 'Cadillac', 'Chevrolet', 'Ferrari', 'Ford', 'Gmc', 'Honda', 'Hyundai',
                'Jaguar Land Rover', 'Jeep', 'Kia', 'Lamborghini', 'Mahindra', 'Maruti Suzuki', 'Mazda', 'Mercedes', 'Mitsubishi',
                'Nissan', 'Peugeot',
                'Porsche', 'Rolls Royce', 'Tata Motors', 'Tesla', 'Toyota', 'Volkswagen', 'Volvo'
            ]
        )

        CC = st.number_input(
            "Engine CC",
        
        )

        horsePower = st.number_input(
            "Horsepower",
            min_value=0
        )

        topSpeed = st.number_input(
            "Top Speed",
            min_value=0
        )
    
        cylinder = int(
            st.selectbox(
                "Cylinders",
                [2, 3, 4, 6, 8, 10, 12]
            )
        )

    with col2:

        oneto100Speed = st.number_input(
            "0-100 km/h Time (seconds)",
            min_value=0.0,
            step=0.1
        )

        seats = int(
            st.selectbox(
                "Seats",
                [1, 2, 3, 4, 5, 7, 8, 9, 12, 15]
            )
        )

        torque = st.number_input(
            "Torque",
            min_value=0
        )

        fuel = st.selectbox(
            "Fuel Type",
            ["Petrol", "Diesel", "Hybrid", "Electric", "CNG", "Hydrogen"]
        )

        Is_Petrol = 0
        Is_Diesel = 0
        Is_Hybrid = 0
        Is_Electric = 0
        Is_CNG = 0
        Is_Hydrogen = 0

        if fuel == "Petrol":
            Is_Petrol = 1

        elif fuel == "Diesel":
            Is_Diesel = 1

        elif fuel == "Hybrid":
            Is_Hybrid = 1

        elif fuel == "Electric":
            Is_Electric = 1

        elif fuel == "CNG":
            Is_CNG = 1

        elif fuel == "Hydrogen":
            Is_Hydrogen = 1


    

        feature_columns = [

            'Acura', 'Aston Martin', 'Audi', 'Bentley', 'Bmw',

            'Bugatti', 'Cadillac', 'Chevrolet', 'Ferrari', 'Ford',

            'Gmc', 'Honda', 'Hyundai', 'Jaguar Land Rover',

            'Jeep', 'Kia', 'Lamborghini', 'Mahindra',

            'Maruti Suzuki', 'Mazda', 'Mercedes', 'Mitsubishi',

            'Nissan', 'Peugeot', 'Porsche', 'Rolls Royce',

            'Tata Motors', 'Tesla', 'Toyota', 'Volkswagen',

            'Volvo', 'CC/Battery Capacity', 'HorsePower',

            'Total Speed', 'Performance(0 - 100 )KM/H',

            'Seats', 'Torque', 'Is_Petrol', 'Is_Diesel',

            'Is_Hybrid', 'Is_Electric', 'Is_CNG',

            'Is_Hydrogen', 'cylinder'

        ]


    if st.button("Predict Price"):
        

        input_data = {col: 0 for col in feature_columns}

        if brand in input_data:

            input_data[brand] = 1


        input_data["CC/Battery Capacity"] = CC

        input_data["HorsePower"] = horsePower

        input_data["Total Speed"] = topSpeed

        input_data["Performance(0 - 100 )KM/H"] = oneto100Speed

        input_data["Seats"] = seats

        input_data["Torque"] = torque

        input_data["cylinder"] = cylinder


        input_data["Is_Petrol"] = Is_Petrol

        input_data["Is_Diesel"] = Is_Diesel

        input_data["Is_Hybrid"] = Is_Hybrid

        input_data["Is_Electric"] = Is_Electric

        input_data["Is_CNG"] = Is_CNG

        input_data["Is_Hydrogen"] = Is_Hydrogen

        input_df = pd.DataFrame([input_data])

        
        prediction = model.predict(input_df)[0];

      
        actual_price = np.exp(prediction)



        lowrange= actual_price*0.87
        highrange = actual_price*1.13

     
        st.markdown(f""" 
                    <div style= "
                    padding = 20px;
                    border-radius = 15px;
                    background:#1E293B;
                    text-align:center;
                    ">
                     <h3>Price Prediction: ${actual_price:,.0f} </h3>

                    </div>
                    """,unsafe_allow_html= True)
        

        st.markdown(f""" 
                    <div style= "
                    padding = 20px;
                    border-radius = 15px;
                    background:#1E293B;
                    text-align:center;
                    ">
                    <h3>Expected Price Range : ${lowrange:,.0f} - ${highrange:,.0f}</h3>
                    </div>
                    """,unsafe_allow_html= True)
with tab2:


    st.header("EDA (Exploratory Data Analysis)", divider="grey") 

    st.subheader("Factors affecting the price most")
    st.image("images/corr_matrix.png",
            caption = "co-relation matrix")
    st.markdown(
        "Features showing the strongest linear relationship with price:<br>"
        "1.HorsePower<br>"
        "2.Highest speed<br>"
        "3.CC of Battery<br>"
        "4.Cylinder <br>" 
        "Fun observation : Seat is actually inversely realated to price , i could be because highly expensive cars usually have 2 seats but big family have more seats but they are cheaper", 


        unsafe_allow_html=True
    )

    st.subheader("Exploring the relation between the HorsePower & Price")

    st.image(
        "images/price_vs_horsePower.png"
    )
    st.markdown(
        "-> we can see how HorsePower influences the price specially after HP>700 , it does represent something close to expontential graph <br>"
        "-> also we can see the extreme outlier , it's mostly a good idea to remove them for better accuracy in real life scenario"
        ,

        unsafe_allow_html=True
    )
    
    st.subheader("Exploring the distribution of the price")

    st.image("images/price.png")
    st.markdown(
        "-> we can see highly right skewed data <br>"
        "-> A model trained on highly skewed data like this can produce poor result because few Outlier will influence the learning of model alot which will make it harder for model to predict for normal range cars<br>"
        "-> I got around 50&percnt; of cross-validated R-sqaured",

        unsafe_allow_html=True
    )

    st.subheader("Exploring the distribution of LogPrice")
    st.image("images/logprice.png")

    st.markdown(
       "-> Applying a logarithmic transformation reduced the skewness of the target variable and produced a distribution closer to normal. This helped the model learn more effectively and improved cross-validation performance. <br>"
       "-> Therefore this model was trained on LogPrice",

        unsafe_allow_html=True
    )

    st.subheader("Key Findings")

    st.markdown(
       "• HorsePower shows the strongest relationship with vehicle price<br>"
        "• Vehicle performance metrics generally correlate positively with price.<br>"

        "• The dataset contains significant high-end outliers. <br>"

        "• Car prices are heavily right-skewed.<br>"

        "• Log transformation improved target distribution and model performance.<br>"

        "• RandomForestRegressor achieved approximately 72% cross-validated R² on the transformed target.",

        unsafe_allow_html=True
    )


with tab3:
    st.header("Metrics of Model" , divider= "grey")

    st.subheader("Model : RandomForestRegressor")

    st.image(
        "images/metrics.png"
    )

    st.markdown(
        "-> cross validated R-squared value for tuned model : 74%  <br>"
        "-> For untuned R-squared : 70% <br>"
        "-> Mean Absolute Error : 12.8k (i forgot to set the random state you if you execute the same code in your system it might deviate a litle) <br> "
        "-> Mean Absolute Error percentage : 13.3%",
        unsafe_allow_html=True
    )

    st.image(
        "images/cross_val.png"
        , caption = "Cross validated R-squared variation"
    )

    st.markdown(
        "-> A consistent pattern was observed that one set was consistently performing bad <br>"
        "-> This could happens because of that set containing alot of variation & outliers <br>"
        "-> Without random state the i observed that sometimes MAE would reach like 20K that was probabaly happening because of this section of data <br> "
        "-> But the overall MAPE always remained in 12-15% ",
        unsafe_allow_html = True
    )

    st.subheader("Model : GradiantBoostingRegressor")

    st.image("images/metrics_boost.png")

    st.markdown(
        "-> Cross validate R-squared mean is 72% <br>"
        "-> Mean Abosolute Error : 22K <br>"
        "-> Root Mean Squared Error : exp(0.257) ≈ 1.29 : 29% <br> "
        "-> Absolute Mean Error Percentage : 18%",

        unsafe_allow_html= True
    )

with tab4: 

    st.header("R-Squared" , divider = "grey")
    
    st.markdown(
        "R-Squared is basically the measure of the how much variance is explained/reduced by the model compared to the variance along mean. <br>",


        unsafe_allow_html = True
    )

    st.image(
        "images/R2.webp",
        caption = " Pic for reference"
    )

    st.markdown(
        "A score of 1.0 means perfect predictions, while 0 means the model performs no better than guessing the average price.<br>"
        "- **R² = 1.0** → Perfect predictions <br> "

        "- **R² = 0.0** → No better than predicting the average car price for every vehicle <br>"

        "- **R² < 0.0** → Worse than simply guessing the average price <br> ",


        unsafe_allow_html = True
    )

    st.header("Cross Validation " , divider = "grey")

    st.image(
        "images/cross_validation.png"
    )

    st.markdown(
        "->In Machine Learning it's a common practice to split the data into training & testing data usually the split is 80-20<br>"
        "->Cross Validation is making sure that no one set is Biased , imagine in 100 data point last 20 ones are really simple to predict & we get like 95%  but for other set it fall to 75% that's why use cross validation <br>"
        "->Cross Validation can split the data into 5 sets(can be changed) of training & testing to see the variation along the whole data set <br> "
        "->The average cross-validation score is a more reliable estimate of real-world performance.",
        unsafe_allow_html = True
    )
    st.info("""
        Our model achieved an average Cross-Validated R² score of 0.74 across 5 folds.
    """)

    st.header("Mean Absolute Error (MAE)" , divider = "grey")

    st.markdown(
        "->On average, the model's predictions differ from actual car prices by about 12.8k<br>"
        "->For a dataset containing vehicles ranging from $20,000 family cars to multi-million-dollar hypercars, this error is reasonable. ",

        unsafe_allow_html= True
    )

    st.header("Mean Absolute Error percentage (MAPE)" , divider = "grey")

    st.markdown(
        "-> Measure of difference of prediction & actual price in percentage <br> "
        "-> In this case MAPE is better metrics because MAE can be heavily influenced by super luxury car <br>"
        "->On average, predictions are off by approximately 13%.<br>"

        "For example:<br>"

        "If a car costs 100,000, the model will typically predict a value between 87,000 and 113,000.",
        unsafe_allow_html= True
    )




   




    
    