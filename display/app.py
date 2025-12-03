from flask import Flask, render_template
import pandas as pd
import os

app = Flask(__name__)

FINAL_PATH = r"C:\Users\rick2\Documents\PPG Project\data\final"

def get_latest_final_csv():
    csv_files = [f for f in os.listdir(FINAL_PATH) if f.endswith(".csv")]
    if not csv_files:
        return None
    latest_file = max([os.path.join(FINAL_PATH, f) for f in csv_files], key=os.path.getmtime)
    return latest_file

@app.route("/")
def index():
    latest_file = get_latest_final_csv()
    if not latest_file:
        return "<h2>No final data found.</h2>"

    df = pd.read_csv(latest_file)
    # Convert dataframe to list of dicts for easier rendering
    records = df.to_dict(orient="records")
    return render_template("index.html", records=records, filename=os.path.basename(latest_file))

if __name__ == "__main__":
    app.run(debug=True)
