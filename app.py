import streamlit as st
import pickle
import pandas as pd
import folium
from streamlit_folium import folium_static
import qrcode
from PIL import Image
import io

# Load the trained model
with open('linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)
# Load the scaler used during training
with open('linear_regression_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
# App public link
APP_URL = "https://machine-learning-model-exfaumy3w5vyei7htlpfxa.streamlit.app/"

# Function to show location on the map
def show_location_on_map(latitude, longitude):
    map = folium.Map(location=[latitude, longitude], zoom_start=14)
    folium.Marker([latitude, longitude], popup=f"Location: {latitude}, {longitude}").add_to(map)
    folium_static(map)


def show_qr_code(link):
    qr = qrcode.make(link)
    buf = io.BytesIO()
    qr.save(buf, format="PNG")
    buf.seek(0)
    image = Image.open(buf)
    st.image(image, caption="📲 Scan to Access the App", width=200)


def get_real_estate_links(latitude, longitude):
    # For simplicity, these are just static links to popular websites selling homes in California
    st.markdown("### 🏡 Real Estate Listings in Your Area (within 50km):")
    st.write("1. [Realtor.com - Homes for Sale in California](https://www.realtor.com/realestateandhomes-search/California)")
    st.write("2. [Zillow - Homes for Sale in California](https://www.zillow.com/ca/homes/)")
    st.write("3. [Redfin - Homes for Sale in California](https://www.redfin.com/CA/homes-for-sale)")
    st.write("4. [Trulia - Homes for Sale in California](https://www.trulia.com/for_sale/California/)")

def app():
    st.title('House Price Prediction ✨')
    
    st.markdown("""
    ###  Do you plan to buy a house in California? 🏠  
    This AI model predicts **California house prices** using machine learning.  
    Just choose your inputs like **location**, **house size**, **rooms**, **Income** and get an instant price estimate!  
    Developed by **Mo. Abdo** ©  
    """)

    # Input features
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
        # Scale the input data using the same scaler used in training
        scaled_input = scaler.transform(input_data)
         # Predict with the scaled input
        predicted_price = model.predict(scaled_input)[0]
        predicted_price = max(predicted_price, 0)
        
        st.markdown(f"💰 Predicted House Price: <h3 style='color: green;'> ${predicted_price * 100000:.2f}</h3>", unsafe_allow_html=True)
        st.write("📍**Location on Map:** ")
        show_location_on_map(Latitude, Longitude)
        get_real_estate_links(Latitude, Longitude)

    st.markdown("---")
    st.markdown("### Scan Me! 📲")
    show_qr_code(APP_URL)

if __name__ == '__main__':
    app()
