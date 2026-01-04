from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# ================= LOAD ARTIFACTS =================
ohe = joblib.load("models/rc_ru_encoding.pkl")
scaler = joblib.load("models/rc_ru_scaler.pkl")

recycle_model = joblib.load("models/rcmodel.pkl")
reuse_model = joblib.load("models/rumodel.pkl")
resale_model = joblib.load("models/rsmodel.pkl")

# ================= COLUMNS =================
categorical_cols = [
    "item_category",
    "material_texture",
    "item_color",
    "approximate_size",
    "condition",
    "item_usage",
    "recycling_symbol",
    "location"
]

numerical_cols = ["approximate_quantity"]

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/ourai")
def ourai():
    return render_template("ourai.html")

# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # 1️⃣ RAW INPUT DF
        df_raw = pd.DataFrame([{
            "item_category": data["item_category"],
            "material_texture": data["material_texture"],
            "item_color": data["item_color"],
            "approximate_size": data["approximate_size"],
            "approximate_quantity": float(data["approximate_quantity"]),
            "condition": data["condition"],
            "item_usage": data["item_usage"],
            "recycling_symbol": data["recycling_symbol"],
            "location": data["location"]
        }])

        # 2️⃣ RESALE MODEL (RAW INPUT ONLY)
        resale = float(resale_model.predict(df_raw)[0])

        # 3️⃣ OHE + SCALING FOR RECYCLE/REUSE
        X_cat = ohe.transform(df_raw[categorical_cols])
        X_cat = pd.DataFrame(
            X_cat.toarray(),
            columns=ohe.get_feature_names_out(categorical_cols)
        )

        X_num = scaler.transform(df_raw[numerical_cols])
        X_num = pd.DataFrame(X_num, columns=numerical_cols)

        X_final = pd.concat([X_num, X_cat], axis=1)

        recycle = float(recycle_model.predict(X_final)[0])
        reuse = float(reuse_model.predict(X_final)[0])

        # 4️⃣ RECOMMENDATION
        scores = {
            "Recycle": round(recycle, 2),
            "Reuse": round(reuse, 2),
            "Resale": round(resale, 2)
        }

        best = max(scores, key=scores.get)

        return jsonify({
            "status": "success",
            "scores": scores,
            "recommendation": best
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)
