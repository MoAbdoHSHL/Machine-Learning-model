import folium

# Latitude and Longitude for the location (e.g., San Francisco)
latitude = 37
longitude = -122
# Create a map centered around the coordinates
map = folium.Map(location=[latitude, longitude], zoom_start=15)

# Add a marker to the map
folium.Marker([latitude, longitude], popup="San Francisco").add_to(map)

# Save the map as an HTML file and open it
map.save("location_map.html")

# Open the map in your browser
import webbrowser
webbrowser.open("location_map.html")
