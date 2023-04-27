import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms


options = ['',"Capsule", "Carpet", "Hazelnut", "Leather", "Metal Nut", "Pill", "Tile", "Wood"]
selected_option = ""

def evaluate(input_img, object):
    if(object == "Capsule"):
        model = torch.load("models/capsule_model.h5")
    elif(object == "Carpet"):
        model = torch.load("models/carpet_model.h5")
    elif(object == "Hazelnut"):
        model = torch.load("models/hazelnut_model.h5")
    elif(object == "Leather"):
        model = torch.load("models/leather_model.h5")
    elif(object == "Metal Nut"):
        model = torch.load("models/metal_nut_model.h5")
    elif(object == "Pill"):
        model = torch.load("models/pill_model.h5")
    elif(object == "Tile"):
        model = torch.load("models/tile_model.h5")
    elif(object == "Wood"):
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
        st.success('Object does not have Defect')
    else:
        st.error('Object has Defects')


def main():
    st.title("Defect Detector")

    st.write("Checks whether given object is Defective or Non-Defective")

    selected_option = st.selectbox("Select an Object", options)

    if selected_option:
        st.write("Object for testing is : ", selected_option)
        uploaded_file = None
        captured_file = None
        st.caption("Capture Image using Camera")
        if st.button("Open Camera"):
            captured_file = st.camera_input("Capture Image")
            st.write("Image captured")
            # captured_file = Image.open(capture)
            if captured_file :
                # st.write("Inside capt file")
                if st.button("evaluate"):
                    # st.write("Inside evaluate")
                    evaluate(captured_file, selected_option)
        st.write("OR")
        uploaded_file = st.file_uploader("Choose an image...", type="png")
        if uploaded_file is not None:
            if st.button("evaluate"):
                evaluate(uploaded_file, selected_option)

if __name__ == "__main__":
    main()
