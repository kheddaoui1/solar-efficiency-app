# File: app.py
from flask import Flask, render_template, request
import xgboost as xgb
import pandas as pd
import plotly.graph_objs as go

app = Flask(__name__)

# Load both models once
model_efficiency = xgb.XGBRegressor()
model_efficiency.load_model("model/solar_model.json")

model_egrid = xgb.XGBRegressor()
model_egrid.load_model("model/solar_model_egrid.json")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    prediction_type = request.form.get('prediction_type') # Get the selected prediction type

    if not file:
        return "No file uploaded", 400

    if prediction_type not in ['efficiency', 'egrid']:
         return "Invalid prediction type selected", 400

    df = pd.read_csv(file)
    required_cols = ["T_Amb", "GlobHor", "WindVel"]
    if not all(col in df.columns for col in required_cols):
        return "Missing required columns: T_Amb, GlobHor, WindVel", 400

    # Select the model based on user input
    if prediction_type == 'efficiency':
        model = model_efficiency
        y_label = "Efficiency"
        line_color = "blue"
    else: # prediction_type == 'egrid'
        model = model_egrid
        y_label = "E_Grid"
        line_color = "red"

    predictions = model.predict(df[required_cols])
    df["Prediction"] = predictions

    # Use Date if available, else index for x-axis
    x_axis = df["Date"] if "Date" in df.columns else df.index

    # Create frames for animation, one point added at each frame
    frames = []
    for i in range(1, len(df) + 1):
        frame = go.Frame(
            data=[go.Scatter(x=x_axis[:i], y=predictions[:i], mode='lines+markers',
                             line=dict(color=line_color, width=2),
                             marker=dict(size=5))]
        )
        frames.append(frame)

    # Initial data - first point or empty
    init_x = x_axis[:1]
    init_y = predictions[:1]

    fig = go.Figure(
        data=[go.Scatter(x=init_x, y=init_y, mode='lines+markers',
                         line=dict(color=line_color, width=2),
                         marker=dict(size=5))],
        layout=go.Layout(
            title=f"Predicted {y_label} Over Time",
            xaxis=dict(title="Date", showgrid=True, gridcolor='#e0e0e0'),
            yaxis=dict(title=y_label, showgrid=True, gridcolor='#e0e0e0'),
            plot_bgcolor='#f9f9f9',
            paper_bgcolor='#ffffff',
            font=dict(family="Arial, sans-serif", size=12, color="#333"),
            margin=dict(l=50, r=50, b=100, t=50, pad=4),
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(label="Play",
                         method="animate",
                         # Corrected args for play/pause functionality
                         args=[None, {"frame": {"duration": 50, "redraw": True},
                                      "mode": "immediate", # Use immediate mode
                                      "fromcurrent": True}]), # Resume from current frame
                    dict(label="Pause",
                         method="animate",
                         # Corrected args for play/pause functionality
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}]) # Stop immediately
                ],
                showactive=False,
                x=0.5, # Center horizontally
                y=-0.2, # Position below the plot
                xanchor="center",
                yanchor="top"
            )]
        ),
        frames=frames
    )

    graph_html = fig.to_html(full_html=False)

    return render_template("result.html", plot=graph_html, prediction_type=y_label)

if __name__ == "__main__":
    app.run(debug=True)
