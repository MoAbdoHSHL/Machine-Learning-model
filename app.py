import streamlit as st
import pickle
import pandas as pd
import folium
from streamlit_folium import folium_static  


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


def show_location_on_map(latitude, longitude):
    # Create a map centered around the provided latitude and longitude
    map = folium.Map(location=[latitude, longitude], zoom_start=14)

    # Add a marker to the map
    folium.Marker([latitude, longitude], popup=f"Location: {latitude}, {longitude}").add_to(map)

    # Display the map using folium_static
    folium_static(map)

def app():
    st.title('House Price Prediction-GPT ✨')
    st.markdown("""
    ###  Do you plan to buy a house in California? 🏠
    This AI model predicts **California house prices** using machine learning.  
    Just choose your inputs like like **location**, **house size**, **rooms**, **Income** and get an instant price estimate!
    Developed by **Mo. Abdo** ©
                """)
    
    MedInc = st.number_input('Median Income (Ex: 8.0 represents $80,000 per year))', value=8.0)
    HouseAge = st.number_input('House Age in years (Ex: 30)', value=30)
    AveRooms = st.number_input('Average number of rooms (Ex: 6)', value=6)
    AveBedrms = st.number_input('Average number of bedrooms (Ex: 1)', value=1)
    Population = st.number_input('Population in the area (Ex: 1000 represents the total number of people in the area)', value=1000)
    AveOccup = st.number_input('Average Occupancy (Ex: 3 represents the average number of people per household)', value=3) 
    Latitude = st.number_input('Latitude (Ex: 34.0 represents the latitude coordinate of the location)', value=34.0)
    Longitude = st.number_input('Longitude (Ex: -118.0 represents the longitude coordinate of the location)', value=-118.0)

    if st.button('Predict House Price'):
        input_data = pd.DataFrame([{
            'MedInc': MedInc,
            'HouseAge': HouseAge,
            'AveRooms': AveRooms,
            'AveBedrms': AveBedrms,
            'Population': Population,
            'AveOccup': AveOccup,
            'Latitude': Latitude,
            'Longitude': Longitude
        }])

        # Predict the price using the loaded model
        predicted_price = model.predict(input_data)[0]
        predicted_price = max(predicted_price, 0)
        st.write(f"💰 Predicted House Price: **${predicted_price * 100000:.2f}** ")
        st.write("📍**Location on Map:** ")
        show_location_on_map(Latitude, Longitude)

if __name__ == '__main__':
    app()
