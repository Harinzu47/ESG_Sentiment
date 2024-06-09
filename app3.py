import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt
from PIL import Image

# Load the tokenizer for bert-base-uncased
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model architecture
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Load the state dictionary into the model
model.load_state_dict(torch.load('model_path.pth', map_location=device))

# Move the model to the appropriate device
model.to(device)

# Function to predict sentiment of a given text
def predict_sentiment(text):
    model.eval()

    # Tokenize and encode the text
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True  # Explicitly activate truncation
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Perform prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    logits = logits.detach().cpu().numpy()

    # Get the predicted label
    predicted_label = logits[0].argmax()

    return predicted_label

# Mapping of predicted labels to sentiment descriptions
sentiment_mapping = {
    0: 'Positif',
    1: 'Negatif',
    2: 'Netral'
}

# Main function for the Streamlit app
def main():
    # Set page config
    st.set_page_config(
        page_title="Aplikasi Analisis Sentimen",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="auto",
    )

    # Add custom CSS for dark and white mode
    def set_theme(theme):
        if theme == "dark":
            st.markdown(
                """
                <style>
                body {
                    color: #fff;
                    background-color: #121212;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
        elif theme == "white":
            st.markdown(
                """
                <style>
                body {
                    color: #256;
                    background-color: #ffffff;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

    # Add theme selector
    theme = st.sidebar.radio("Select theme", ["dark", "white"], index=0)
    set_theme(theme)

    # Load and resize logo image
    image_path = "Logofix.png"
    img = Image.open(image_path)
    img.thumbnail((300, 300))  # Resize the image to fit within 800x800 pixels

    # Display logo image
    st.image(img)
    st.title("Dashboard Analisis Sentimen")

    st.write("Ini adalah aplikasi untuk analisis sentimen dengan Topik ESG")

    # File upload
    uploaded_file = st.file_uploader("Unggah file CSV untuk analisis sentimen dalam jumlah banyak", type=["csv"])

    if uploaded_file:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file, on_bad_lines='skip')

        # Check if the expected column exists
        if 'Text' not in df.columns:
            st.write("File CSV harus memiliki kolom 'text'")
        else:
            # Predict sentiment for each text
            df['Prediksi Sentimen'] = df['Text'].apply(lambda x: sentiment_mapping[predict_sentiment(x)])

            # Display the results
            st.write("Hasil Prediksi Sentimen:")
            st.write(df)

            # Plot the pie chart
            sentiment_counts = df['Prediksi Sentimen'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

            st.pyplot(fig)

    # Single text input
    user_input = st.text_area("Masukkan teks untuk analisis sentimen:")

    if st.button("Analisis"):
        if user_input:
            # Predict sentiment
            predicted_sentiment = predict_sentiment(user_input)
            predicted_sentiment_description = sentiment_mapping[predicted_sentiment]

            # Display the results
            results_df = pd.DataFrame({
                'Teks Sentimen': [user_input],
                'Prediksi Sentimen': [predicted_sentiment_description]
            })

            st.write("Hasil Prediksi Sentimen:")
            st.write(results_df)
        else:
            st.write("Silakan masukkan teks untuk analisis.")

if __name__ == "__main__":
    main()
