2. Required Model Assets
Ensure the following pretrained weight files are present in your root directory:

best_tomato_model_V2.keras (Disease Classification weights)

yolo_model.pt (Tomato Fruit Detection weights)

3. Run the Server
Launch the local Uvicorn instance with active reloading enabled:

Bash
python -m uvicorn main:app --reload
Once initialized, head to http://127.0.0.1:8000/docs to test the system live inside the Swagger Interactive UI.

📄 API Response Examples
Leaf Scan Response
JSON
{
  "analysis_type": "leaf",
  "status": "Tomato___Early_blight",
  "confidence": "94.2%",
  "treatment": "Prune infected lower leaves. Apply Mancozeb or Copper-based fungicides.",
  "is_healthy": false
}
Fruit Scan Response
JSON
{
  "analysis_type": "fruit",
  "total_fruits": 7,
  "harvest_status": "Turning / Almost Ready",
  "estimated_days_to_harvest": "3 - 5 Days"
}
