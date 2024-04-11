import streamlit as st
import os
from PIL import Image
import io
from knowmad import tell_answer_from_uploaded_material,give_video_explanation
# Function to upload documents and store them locally
counter=-1
def upload_documents():
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader("Upload your files:", type=['txt', 'pdf'], accept_multiple_files=True)
    
    if uploaded_files:
        for file in uploaded_files:
            with open(os.path.join("training", file.name), "wb") as f:
                f.write(file.getbuffer())
        st.success("Files uploaded successfully!")

# Main function to create the web app
def main():
    st.title("KNOWMAD")
    text_input = st.text_input("Enter some text:")

    # Button to upload documents
    if st.button("Upload Documents"):
        upload_documents()

    # Button to run main pipeline
    if st.button("Tell answer from uploaded material"):
        result=tell_answer_from_uploaded_material(text_input)
        st.text_area("Output:", value=result, height=100)

    uploaded_image = st.file_uploader("Upload an image:", type=['jpg', 'png', 'jpeg'])
    if uploaded_image is not None:
        # Convert the uploaded file to a PIL Image
        pil_image = Image.open(io.BytesIO(uploaded_image.read()))
        
        # Now you can work with the PIL Image
        st.image(pil_image, caption="Uploaded Image")
    # Button to run next pipeline
    if st.button("Video Explanation"):
        global counter
        counter+=1
        result=give_video_explanation(text_input,counter,pil_image)
        st.text_area("Output:", value=result, height=100)

if __name__ == "__main__":
    main()
