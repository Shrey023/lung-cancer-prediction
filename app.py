import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import json
import timm
import requests

# Load label mapping
with open("label_mapping.json", "r") as f:
    label_mapping = json.load(f)

raw_class_names = list(label_mapping.keys())

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load modelN
def load_model():
    model = timm.create_model('efficientnet_b4', pretrained=False)
    num_classes = len(raw_class_names)
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    model.load_state_dict(torch.load("lung_cancer_model_effb4.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model

model = load_model()

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Gemini API setup
GEMINI_API_KEY = "AIzaSyBmf62BlwWARrVjDKMgj4JJ_rTRKhfQaqw"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

def get_gemini_suggestion(cancer_type, stage_num):
    stage_desc = {
        0: "Normal",
        1: "Stage 1",
        2: "Stage 2",
        3: "Stage 3",
        4: "Stage 4"
    }.get(stage_num, "Unknown Stage")

    prompt = f"""
    You are a helpful assistant providing general educational information.
    A patient has been diagnosed with {cancer_type} ({stage_desc}).
    Provide a short and concise overview of:
    - Diagnosis meaning
    - Common tests
    - Treatment options
    - Lifestyle advice
    Keep each point brief. Do not provide personal medical advice.
    """

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    try:
        response = requests.post(
            GEMINI_API_URL,
            headers={"Content-Type": "application/json"},
            json=payload
        )
        data = response.json()

        if "candidates" in data and len(data["candidates"]) > 0:
            suggestion = data["candidates"][0]["content"]["parts"][0]["text"]

            # Truncate long responses
            MAX_CHAR = 500
            if len(suggestion) > MAX_CHAR:
                suggestion = suggestion[:MAX_CHAR] + "...\n\n(Shortened for clarity)"

            return suggestion
        else:
            return "⚠️ No suggestion returned from Gemini."
    except Exception as e:
        return f"❌ Error fetching suggestion: {str(e)}"

# Streamlit UI
st.title("🫁 Lung Cancer Prediction")
st.markdown("Upload a chest X-ray or CT scan image for classification.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        conf, predicted_idx = torch.max(probabilities, 0)

    predicted_raw_label = raw_class_names[predicted_idx.item()]
    cancer_info = label_mapping[predicted_raw_label]
    confidence = conf.item() * 100

    st.subheader("🔍 Prediction:")
    st.write(f"**Type:** {cancer_info['type']}")

    stage_num = cancer_info.get("stage_num", 0)
    stage_desc = {
        0: "Normal",
        1: "Stage 1",
        2: "Stage 2",
        3: "Stage 3",
        4: "Stage 4"
    }.get(stage_num, "Unknown Stage")

    if stage_num == 1:
        st.markdown('<span style="color:green;">**Stage: 1 (Early)**</span>', unsafe_allow_html=True)
    elif stage_num == 2:
        st.markdown('<span style="color:orange;">**Stage: 2 (Moderate)**</span>', unsafe_allow_html=True)
    elif stage_num >= 3:
        st.markdown('<span style="color:red;">**Stage: 3+ (Advanced)**</span>', unsafe_allow_html=True)
    else:
        st.write("**Stage: N/A**")

    st.subheader(f"📊 Confidence: {confidence:.2f}%")

    if stage_num > 0:
        st.markdown("---")
        st.subheader("💡 Medical Suggestion:")

        suggestion = get_gemini_suggestion(cancer_info['type'], stage_num)
        st.markdown(suggestion)

        # === Chatbot Section ===
        st.markdown("---")
        st.subheader("🤖 Ask the AI Assistant")
        st.markdown("💬 *Examples: 'What does Stage 1 mean?', 'Can this be cured?', 'What should I do next?'*")

        # Initialize session state for chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        for role, text in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(text)

        # Chat input
        user_input = st.chat_input("Ask something related to your result...")

        if user_input:
            # Add user message
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.chat_history.append(("user", user_input))

            # Build context based on prediction
            context = f"""
You were shown results indicating a diagnosis of {cancer_info['type']} ({stage_desc}).
The AI previously suggested:
"{suggestion}"

User now asked: {user_input}
"""

            full_prompt = f"""
You are a helpful medical assistant providing general educational information.

{context}

Provide a short, relevant answer based on the above. Always remind users to consult a doctor.
Do not give personal medical advice.
"""

            payload = {
                "contents": [{"parts": [{"text": full_prompt}]}]
            }

            try:
                response = requests.post(GEMINI_API_URL, json=payload, headers={"Content-Type": "application/json"})
                data = response.json()

                if "candidates" in data and len(data["candidates"]) > 0:
                    ai_response = data["candidates"][0]["content"]["parts"][0]["text"]
                    with st.chat_message("assistant"):
                        st.markdown(ai_response)
                    st.session_state.chat_history.append(("assistant", ai_response))
                else:
                    st.warning("⚠️ No response from assistant.")
            except Exception as e:
                st.error(f"❌ Error fetching response: {str(e)}")

        # Final disclaimer
        st.markdown("""
        ⚠️ Note: This is general educational information only. 
        Always consult a licensed healthcare provider for medical concerns.
        """)