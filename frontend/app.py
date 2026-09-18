import streamlit as st
import requests
import json

API_BASE_URL = "http://127.0.0.1:5000"

st.set_page_config(
    page_title="Product category classifier",
    page_icon="https://img.icons8.com/?size=100&id=GthReUBAkauZ&format=png&color=000000",
    layout="wide"
)

st.title("Product Classifier")

with st.sidebar:
    st.header("server settings")
    api_url = st.text_input("Flask API URL",value=API_BASE_URL)

    st.subheader("Server check")
    if st.button("check API status (GET /)",use_container_width=True):
        try:
            res = requests.get(f"{api_url}/",timeout=5)
            if res.status_code==200:
                st.success(f"server res:{res.text}")
            else:
                st.error(f"http error {res.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("Flask app is not running")
        except Exception as e:
            st.error(f"Error:{str(e)}")

col1, col2 = st.columns([1,1],gap="large")

with col1:
    st.subheader("1. Upload Image")
    uploaded_file = st.file_uploader(
        "choose an image", type=["jpg","jpeg","png"]
    )
    if uploaded_file is not None:
        st.image(uploaded_file,caption="Preview",use_container_width=True)

with col2:
    st.subheader("2. Define Categories")
    default_categories = "running shoes, leather shoes, wristwatch, office desk"
    category_input = st.text_area("categories",value=default_categories, height=120)

    st.subheader("3. Classify")
    predict_btn = st.button("Run prediction",type="primary",use_container_width=True)

if predict_btn:
    if uploaded_file is None:
        st.warning("Please upload an image first")
    elif not category_input.strip():
        st.warning("Plese enter atleast one category label.")
    else:
        with st.spinner("Classifying Image..."):
            try:
                files={
                    "image":(uploaded_file.name,uploaded_file.getvalue(),uploaded_file.type)
                }
                data = {
                    "categories":category_input.strip()
                }

                response = requests.post(f"{api_url}/classify",files=files,data=data,timeout=30)

                if response.status_code == 200:
                    payload = response.json()
                    st.success("Classification successful!")

                    result_data = payload.get("result",{})

                    if isinstance(result_data, dict):
                        pred_label = result_data.get("predicted_category")
                        confidence = result_data.get("top_confidence")
                        all_preds = result_data.get("all_predictions")

                    if pred_label:
                        conf_text = f"({confidence*100:.2f}%)" if confidence is not None else ""
                        st.subheader(f"Top Match:`{pred_label}`{conf_text}")

                    if all_preds:
                        st.markdown("All Prediction:")
                        for item in all_preds:
                            label = item.get("label","")
                            score = float(item.get("confidence",0.0))
                            st.write(f"{label}: {score*100:.2f}%")
                            st.progress(min(max(score,0.0),1.0))

                    with st.expander("view Full json response"):
                        st.json(payload)

                else:
                    st.error(f"Error {response.status_code}: request failed")
                    st.code(response.text)

            except requests.exceptions.ConnectionError:
                st.error("Flask API is not running")
            except Exception as e:
                st.error(f"error {str(e)}")
