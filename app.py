import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification



st.set_page_config(
    page_title="Data Hunters",
    page_icon="src\Huntersb.png",  # Reemplaza con la ruta de tu logo
)
# Cargar el tokenizador y el modelo entrenado
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=5)  # Ajusta num_labels según tus clases

# Título y descripción de la aplicación
st.title('Calificación de comentarios con Data Hunters')
st.write('Ingrese un comentario para obtener una predicción de sentimiento.')

# Área de entrada de texto
comentario = st.text_input('Ingrese su comentario:')

# Botón para realizar la predicción
if st.button('Predecir Sentimiento'):
    comentario_encoded = tokenizer.encode_plus(comentario, truncation=True, padding=True, return_tensors='pt')
    with torch.no_grad():
        logits = model(**comentario_encoded)[0]
    predicción = torch.argmax(logits, dim=1).item()
    st.write(f'Predicción de Sentimiento: {predicción + 1} estrellas')



comment = st.text_area("Ingresa tu comentario:")
if st.button("Generar Recomendaciones"):
    # Codifica el comentario y realiza la inferencia
    comment_encoded = tokenizer([comment], truncation=True, padding=True, return_tensors='pt')
    with torch.no_grad():
        logits = model(**comment_encoded)[0]
    prediction = torch.argmax(logits, dim=1)[0].item() + 1

    # Genera una recomendación en base a la predicción
    if prediction == 1:
        st.write("Recomendación para Mejorar: Mejorar la calidad del servicio al cliente.")
    elif prediction == 5:
        st.write("Recomendación para Mantener: Sigue proporcionando un excelente servicio y calidad.")
    else:
        st.write("Recomendación para Considerar: Evalúa posibles mejoras en la experiencia del cliente.")
