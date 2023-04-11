import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms


options = ['',"capsule", "carpet", "hazelnut", "leather", "metal_nut", "pill", "tile", "wood"]
selected_option = ""

def evaluate(input_img, object):
    if(object == "capsule"):
        model = torch.load("models/capsule_model.h5")
    elif(object == "carpet"):
        model = torch.load("models/carpet_model.h5")
    elif(object == "hazelnut"):
        model = torch.load("models/hazelnut_model.h5")
    elif(object == "leather"):
        model = torch.load("models/leather_model.h5")
    elif(object == "metal_nut"):
        model = torch.load("models/metal_nut_model.h5")
    elif(object == "pill"):
        model = torch.load("models/pill_model.h5")
    elif(object == "tile"):
        model = torch.load("models/tile_model.h5")
    elif(object == "wood"):
        model = torch.load("models/wood_model.h5")
    else:
        return

    # Preprocess the image
    img = Image.open(input_img)
    img = transforms.Resize((224, 224))(img)
    st.image(img, caption="Uploaded image")

    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

    # Pass the image through the model and get the predictions
    with torch.no_grad():
        preds = model.forward(img.unsqueeze(0))

    # Analyze the output to determine whether the image is good or bad
    if preds[0][0] > 0.5:
        st.write('This is a good image.')
    else:
        st.write('This is a bad image.')


def main():
    st.title("Good Checker")

    st.write("It will check if the given object in the image is defective of not")

    selected_option = st.selectbox("Select object to test", options)

    if selected_option:
        st.write("Object for testing is : ", selected_option)
        uploaded_file = None

        uploaded_file = st.file_uploader("Choose an image...", type="png")
        if uploaded_file is not None:
            # image = Image.open(uploaded_file)
            # image = image.resize((300, 300))
            # st.image(image, caption="Uploaded image")

            # # Load the model from an H5 file
            # if(selected_option == "capsule"):
            #     model = torch.load("models/capsule_model.h5")
            # elif(selected_option == "carpet"):
            #     model = torch.load("models/carpet_model.h5")
            # elif(selected_option == "hazelnut"):
            #     model = torch.load("models/hazelnut_model.h5")
            # elif(selected_option == "leather"):
            #     model = torch.load("models/leather_model.h5")
            # elif(selected_option == "metal_nut"):
            #     model = torch.load("models/metal_nut_model.h5")
            # elif(selected_option == "pill"):
            #     model = torch.load("models/pill_model.h5")
            # elif(selected_option == "tile"):
            #     model = torch.load("models/tile_model.h5")
            # elif(selected_option == "wood"):
            #     model = torch.load("models/wood_model.h5")
            # else:
            #     return

            # # Preprocess the image
            # img = Image.open(uploaded_file)
            # img = transforms.Resize((224, 224))(img)
            # img = transforms.ToTensor()(img)
            # img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

            # # Pass the image through the model and get the predictions
            # with torch.no_grad():
            #     preds = model.forward(img.unsqueeze(0))

            # # Analyze the output to determine whether the image is good or bad
            # if preds[0][0] > 0.5:
            #     st.write('This is a good image.')
            # else:
            #     st.write('This is a bad image.')

            if st.button("evaluate"):
                evaluate(uploaded_file, selected_option)

if __name__ == "__main__":
    main()