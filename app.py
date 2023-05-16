import streamlit as st

# import required packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import cv2

# import the packages we made
from utils import feature_engineering as fe
from models import predict

# set current working space
dir = os.getcwd()

# Set the title and a brief description for the app
st.title("Satellite Image Methane Detection")
st.subheader("A QuantumBlack Hackathon Project")

st.markdown("---")

# Define the team information
team_name = 'Team Adeptus Mechanicus'
team_members = ['Yifan WANG', 'Dayu LIU', 'Yaqi CHEN', 'Peizhen CHEN', 'Xiangying CHEN']
github_link = "https://github.com/liudayu991114/QBHackathon"

# Display team information
st.header("Team Information")
st.subheader(team_name)
st.write("Team members:")
for member in team_members:
    st.write(f"- {member}")
st.write(f"[GitHub Link]({github_link})")

st.markdown("---")

# working area of the app
st.header("Image Analysis")

# First part, single image analysis
st.subheader("Single Image Analysis")
st.write("Simply upload an image and the application will analyze it for you.")
st.write("You can upload an image of any size and any format. It can be grayscale or RGB.")
uploaded_file = st.file_uploader("Upload an image", type = ["jpg", "jpeg", "png", "tif"])
# process after the file is loaded
if uploaded_file is not None:
    st.write("File received. Analyzing...")
    # preprocess the image
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    image = fe.fileprocess(image)
    # display the preprocessed image
    st.write("The preprocessed image is: ")
    st.image(image, channels = "GRAY")

    st.markdown("---")

    # predict using ensemble methods
    st.write("Multi-model predicting in progress... It may take some time...")
    prob_1 = predict.predict(image, dir = dir, model_choice = 'resnet34', prob = True)
    prob_2 = predict.predict(image, dir = dir, model_choice = 'resnet18', prob = True)
    prob_3 = predict.predict(image, dir = dir, model_choice = 'resnet50', prob = True)
    prob_4 = predict.predict(image, dir = dir, model_choice = 'alexnet', prob = True)
    prob_5 = predict.predict(image, dir = dir, model_choice = 'mobilenet', prob = True)
    prob = 0.35 * np.array(prob_1) + 0.35 * np.array(prob_2) + 0.1 * np.array(prob_3) + 0.1 * np.array(prob_4) + 0.1 * np.array(prob_5)
    prediction = np.argmax(prob)
    # show the prediction
    st.write("The prediction made by ensemble method is: ", prediction)
    if prediction == 1: # show heat-map of the grad-cam
        st.write("Methane Detected!")
        heatmap = predict.heat_map(image, dir = dir)
        st.image(heatmap, channels = "RGB")
    else:
        st.write("Methane Not Detected.")

st.markdown("---")

# Second part, multi-image analysis
st.subheader("Multi-Image Analysis")
st.write("Upload your metadata.csv")
data = st.file_uploader("Upload metadata.csv", type = "csv")

if data is not None:
    st.write("File received. Analyzing...")
    # read the csv and process
    df = pd.read_csv(data)
    df['date'] = pd.to_datetime(df['date'], format = '%Y%m%d')
    files = list(df['path'])
    files = [dir + '/data/' + path + '.tif' for path in files]
    # preprocess the images and predict using emsemble methods
    images = [fe.preprocess(file) for file in files]
    prob_1 = predict.predict(images, dir = dir, model_choice = 'resnet34', prob = True)
    prob_2 = predict.predict(images, dir = dir, model_choice = 'resnet18', prob = True)
    prob_3 = predict.predict(images, dir = dir, model_choice = 'resnet50', prob = True)
    prob_4 = predict.predict(images, dir = dir, model_choice = 'alexnet', prob = True)
    prob_5 = predict.predict(images, dir = dir, model_choice = 'mobilenet', prob = True)
    prob = 0.35 * np.array(prob_1) + 0.35 * np.array(prob_2) + 0.1 * np.array(prob_3) + 0.1 * np.array(prob_4) + 0.1 * np.array(prob_5)
    labels = [np.argmax(arr) for arr in prob]
    df['label'] = labels
    # print some information
    st.write("Total amount of images: ", len(labels))
    label_counts = df['label'].value_counts()
    fig, ax = plt.subplots(figsize = (6, 6))
    ax.set_title("Label Proportions")
    ax.pie(label_counts, labels = label_counts.index, autopct = '%1.1f%%', startangle = 90)
    ax.axis('equal')
    st.pyplot(fig)

    st.markdown("---")

    # Select month to display
    selected_month = st.selectbox("Select a month", sorted(df['date'].dt.strftime('%Y-%m').unique()))
    filtered_df = df[df['date'].dt.strftime('%Y-%m').str.startswith(selected_month)]

    # Plot dots on the world map using geopandas
    st.write("Map:")
    geometry = [Point(lon, lat) for lon, lat in zip(filtered_df['lon'], filtered_df['lat'])]
    gdf = gpd.GeoDataFrame(filtered_df, geometry = geometry)
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    fig, ax = plt.subplots(figsize = (10, 6))
    world.plot(ax = ax, color = 'lightgray', edgecolor = 'black')
    gdf.plot(ax = ax, markersize = 15, column = 'label', cmap = 'RdYlBu_r', legend = True)
    ax.set_title("World Distribution of Images")
    st.pyplot(fig)

st.markdown("---")

# Add a button to reset the page
if st.button("Reset Page"):
    st.experimental_rerun()