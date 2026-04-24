# 🌿 Plant Doctor AI (Plant AI Assistant)

Plant Doctor AI is a **100% offline, multi-agent artificial intelligence framework** for extremely accurate and robust plant disease diagnostics. Developed with agricultural priorities in mind, the system integrates deep learning computer vision, local Vision-Language Models (VLM), localized voice synthesis, and multi-lingual Large Language Models (LLM) to assist farmers offline.

## 🚀 Key Features

*   **Offline-First Architecture**: No API keys or internet connection required. Runs 100% locally on your machine edge-device utilizing local resources.
*   **Dual-Layer Computer Vision**: 
    *   **CNN Backend**: Pre-trained convolutional neural network (Keras/TensorFlow) optimized for 19 specific crop diseases (PlantVillage).
    *   **VLM Validation Layer**: Local *Llava* Vision-Language model integrations for highly rigorous misclassification verification and general visual reasoning reasoning.
*   **Multilingual Voice & Text**: Fluid conversational streaming UI in **English, Hindi, and Punjabi**. Backed by an offline LLM (like *Mistral*) and localized Text-to-Speech engines.
*   **Lesion Localization (Grad-CAM)**: Pixel-perfect highlighting organically pinpoints the exact infected regions directly onto user-uploaded leaf photos.
*   **Advanced Crop Intelligence**:
    *   Estimates severity mapping (Low, Moderate, High/Severe).
    *   Calculates relative *Leaf Age* alongside Chlorophyll extraction.
    *   Organic and Chemical treatment recommendations.
*   **Plant Progress Tracker**: SQLite-powered tracking dashboards to monitor historical scans, field histories, severity trends, and treatment outcomes.
*   **Multi-Interface**: Supports both a comprehensive Streamlit UI Web Application and a full-featured standalone FastAPI backend.

---

## 🛠 Tech Stack

*   **Frontend UI**: [Streamlit](https://streamlit.io/) with rich bespoke CSS styling, custom interactive chat pills, and browser Voice integration.
*   **Backend Server**: [FastAPI](https://fastapi.tiangolo.com/) providing secure local REST API endpoints.
*   **Deep Learning (CV)**: TensorFlow & Keras.
*   **Local Generative AI Agents**: Ollama (for Llama/Mistral/Llava layers).
*   **Computer Vision Utils**: OpenCV (CV2), PIL (Pillow), NumPy.
*   **Data Logistics**: Python `sqlite3` for history tracking.

---

## 📂 Project Structure

```text
plant_ai_assistant/
├── app.py                   # Main Streamlit Frontend Chat and Tracker UI
├── main_fastapi.py          # Standalone FastAPI backend server
├── train_model.py           # Script to train/fine-tune the CNN Plant Dataset model
├── app/modules/
│   ├── cloud_vision.py      # Local VLM (Llava) verification handler
│   ├── leaf_detector.py     # Initial visual filter to check if image is a leaf
│   ├── image_quality.py     # Check glare, blur, and lighting conditions
│   ├── leaf_age_detector.py # Analyze aging, chlorophyll breakdown
│   ├── severity_estimator.py# Analyze proportion of lesion surface areas
│   ├── knowledge_base.py    # Localized offline LLM retrieval for user interactions
│   ├── voice_engine.py      # Cross-platform / Multi-lingual TTS handlers
│   └── db_manager.py        # SQLite functions for progress tracker & histories
├── dataset/                 # Raw and processed PlantVillage imagery (ignored in git)
├── model/                   # Output storage for the .h5 model and class matrices
└── ...                      # Utility scripts (testing, debugging, dataset cleaning)
```

---

## ⚙️ Setup & Installation

**Prerequisites:**
*   Python 3.9+
*   Node.js (Optional for some web compilation)
*   [Ollama](https://ollama.ai/) installed locally.
*   TensorFlow and OpenCV compatible device.

**1. Clone the repository:**
```bash
git clone https://github.com/your-username/plant_ai_assistant.git
cd plant_ai_assistant
```

**2. Install Python Dependencies:**
```bash
pip install -r requirements.txt
# (Alternatively install manually: streamlit fastapi uvicorn tensorflow opencv-python pillow ... )
```

**3. Prepare Offline Language Models (via Ollama):**
Ensure Ollama is running, then pull the necessary models for conversational and vision capabilities:
```bash
ollama pull mistral 
ollama pull llava
```

**4. Train or Place the Model:**
*   Place your pre-trained `plant_disease_model.h5` inside the `model/` directory alongside `class_names.json`.
*   Alternatively, download the PlantVillage dataset into `dataset/PlantVillage` and run `python train_model.py` to generate the `.h5` model file.

---

## 🖥 Usage

You can run the system in two modes:

### 1. Interactive Web Application (Streamlit)
Launch the beautiful user-facing dashboard which features chat interface, voice synthesis, multi-lang toggle, and field progress tracker.
```bash
streamlit run app.py
```
> Access at: `http://localhost:8501`

### 2. Standalone REST API (FastAPI)
Run the backend server if you wish to connect a mobile application or another custom frontend service.
```bash
python -m uvicorn main_fastapi:app --host 0.0.0.0 --port 8000 --reload
```
> Access Swagger UI at: `http://localhost:8000/docs`

---

## 🌐 Localization Support
To make full use of the localized voice synthesis natively:
- **Windows**: Add **Hindi** (`hi-IN`) and **Punjabi** (`pa-IN`) speech packages via *Settings > Time & Language > Speech*.
- **macOS/Linux**: Most browser speech synthesis engines cover these fallback packages automatically. Chrome is recommended for the best microphone input fidelity.

---

## 🤝 Contribution
Feel free to open issues, submit pull requests, or fork the repository. For model alterations, please ensure your labels map properly out of the dataset dictionaries.
