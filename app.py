import streamlit as st
import pickle
import pandas as pd
import folium
from streamlit_folium import folium_static  # Import folium_static for map display

# Load your pre-trained model (saved as 'model.pkl' previously)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to display the location on the map within Streamlit
def show_location_on_map(latitude, longitude):
    # Create a map centered around the provided latitude and longitude
    map = folium.Map(location=[latitude, longitude], zoom_start=15)

    # Add a marker to the map
    folium.Marker([latitude, longitude], popup=f"Location: {latitude}, {longitude}").add_to(map)

    # Display the map using folium_static
    folium_static(map)

# Create the Streamlit app interface
def app():
    st.title('California House Price Prediction  Mo.Abdo-GPT ✨')

    # Input fields for house features
    MedInc = st.number_input('Median Income (Ex: 8.0)', value=8.0)
    HouseAge = st.number_input('House Age in years (Ex: 30)', value=30)
    AveRooms = st.number_input('Average number of rooms (Ex: 6)', value=6)
    AveBedrms = st.number_input('Average number of bedrooms (Ex: 1)', value=1)
    Population = st.number_input('Population in the area (Ex: 1000)', value=1000)
    AveOccup = st.number_input('Average Occupancy (Ex: 3)', value=3)
    Latitude = st.number_input('Latitude (Ex: 34.0)', value=34.0)
    Longitude = st.number_input('Longitude (Ex: -118.0)', value=-118.0)

    # Button to make predictions
    if st.button('Predict House Price'):
        # Create the input data for prediction
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

        # Display the result
        st.write(f"💰 Predicted House Price: ${predicted_price * 100000:.2f}")

        # Show the location map when the user provides latitude and longitude
        show_location_on_map(Latitude, Longitude)

# Run the app
if __name__ == '__main__':
    app()
