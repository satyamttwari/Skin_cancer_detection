import requests
import streamlit as st
from streamlit_lottie import st_lottie
import PIL
import tensorflow as tf

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def load_model():
    try:
        model = tf.keras.applications.InceptionV3(weights='imagenet')
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

def preprocess_image(img):
    img = img.resize((299, 299))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    return img_array

def predict_skin_cancer(image, model):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class = tf.keras.applications.inception_v3.decode_predictions(predictions, top=1)[0][0][1]
    return predicted_class

st.set_page_config(
    page_title="Skin Cancer",
    page_icon="♋",
    layout="wide",
    initial_sidebar_state="expanded",
)

lottie_health = load_lottieurl(
    "https://assets2.lottiefiles.com/packages/lf20_5njp3vgg.json"
)
lottie_welcome = load_lottieurl(
    "https://assets1.lottiefiles.com/packages/lf20_puciaact.json"
)
lottie_healthy = load_lottieurl(
    "https://assets10.lottiefiles.com/packages/lf20_x1gjdldd.json"
)

st.title("Welcome to team Diagnose!")
st_lottie(lottie_welcome, height=300, key="welcome")

st.markdown("## Article Introduction")
st.markdown("""
            Definition:
Skin cancer refers to the abnormal growth of skin cells, typically due to prolonged exposure to ultraviolet (UV) radiation from the sun or other sources. There are several types of skin cancer, with varying degrees of severity.

Types of Skin Cancer:

Basal Cell Carcinoma (BCC):

The most common type of skin cancer.
Develops in the basal cells of the skin's outermost layer.
Usually appears as a flesh-colored, pearl-like bump or a pinkish patch of skin.
Rarely metastasizes but can cause disfigurement if left untreated.
Squamous Cell Carcinoma (SCC):

Develops in the squamous cells, which make up the skin's upper layers.
Often appears as a firm, red nodule or a flat, scaly lesion.
May metastasize to other parts of the body if left untreated.
Melanoma:

The most serious type of skin cancer.
Develops in the melanocytes, the cells that produce melanin (pigment).
Typically appears as a new spot or an existing mole that changes in size, shape, or color.
Can metastasize to other organs and tissues if not detected and treated early.
Risk Factors:

Excessive Sun Exposure: Prolonged exposure to UV radiation from the sun increases the risk of developing skin cancer.
Fair Skin: People with fair skin, light hair, and blue or green eyes are at higher risk.
History of Sunburns: Multiple instances of sunburns, especially during childhood, increase the risk.
Family History: A family history of skin cancer can predispose individuals to the disease.
Weakened Immune System: Conditions that weaken the immune system, such as HIV/AIDS or organ transplantation, increase susceptibility.
Exposure to Certain Substances: Exposure to certain chemicals, such as arsenic, can increase the risk of developing skin cancer.
Prevention:

Sun Protection: Use sunscreen with a high SPF, wear protective clothing, seek shade, and avoid outdoor activities during peak sun hours.
Regular Skin Checks: Perform regular self-examinations to check for any changes in moles, freckles, or skin lesions.
Avoid Tanning Beds: Tanning beds emit UV radiation, increasing the risk of skin cancer.
Healthy Lifestyle: Maintain a healthy diet, exercise regularly, and avoid smoking to reduce overall cancer risk.
Early Detection:

Early detection of skin cancer is crucial for successful treatment.
Look for changes in the size, shape, color, or texture of moles or skin lesions.
Consult a dermatologist if you notice any suspicious skin changes or if you have risk factors for skin cancer.
Treatment:

Treatment options for skin cancer depend on the type, size, and location of the tumor, as well as the individual's overall health.
Common treatment modalities include surgical excision, Mohs surgery, cryotherapy, radiation therapy, chemotherapy, immunotherapy, and targeted therapy.
Overall, skin cancer is highly preventable and treatable when detected early. By practicing sun safety measures, performing regular skin checks, and seeking prompt medical attention for any suspicious skin changes, individuals can reduce their risk of developing skin cancer and improve their chances of successful treatment.
            
           
            """)
            
st.markdown("[Learn More >](https://www.researchgate.net/publication/356093241_Characteristics_of_publicly_available_skin_cancer_image_datasets_a_systematic_review)")




# ... (rest of your introduction code)

# Skin Cancer Prediction Section
# st.header("Skin Cancer Detection")
# pic = st.file_uploader(
#     label="Upload a picture",
#     type=["png", "jpg", "jpeg"],
#     accept_multiple_files=False,
#     help="Upload a picture of your skin to get a diagnosis",
# )

# if st.button("Predict"):
#     if pic:
#         st.header("Results")

#         cols = st.columns([1, 2])
#         with cols[0]:
#             st.image(pic, caption=pic.name, use_column_width=True)

#         with cols[1]:
#             model = load_model()

#             if model:
#                 with st.spinner("Predicting..."):
#                     img = PIL.Image.open(pic)
#                     predicted_class = predict_skin_cancer(img, model)

#                     st.write(f"**Prediction:** `{predicted_class}`")
#             else:
#                 st.error("Model could not be loaded. Please check the model file.")

#         st.warning(
#             ":warning: This is not a medical diagnosis. Please consult a doctor for a professional diagnosis."
#         )
#     else:
#         st.error("Please upload an image")
