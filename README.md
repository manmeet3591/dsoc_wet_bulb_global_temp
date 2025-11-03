# dsoc_wet_bulb_global_temp

# Kentucky Mesonet — Stations Map (Streamlit)

A simple Streamlit app that visualizes Kentucky Mesonet stations statewide and a zoomed view for a selected county (default: Warren). Data is read live via the Mesonet's static manifest.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
