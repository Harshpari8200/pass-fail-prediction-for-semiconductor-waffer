import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Data from the provided JSON
features = ["Surface_Defect_Count", "Pattern_Alignment_Error_nm", "Resistance_Ohms", 
           "Surface_Roughness_nm", "Temperature_C", "Voltage_V", "Wafer_Thickness_um", 
           "Chemical_Concentration_ppm", "Process_Time_min"]
importance = [0.497767, 0.187451, 0.147021, 0.080876, 0.048535, 0.018577, 0.009970, 0.007615, 0.002188]

# Create DataFrame and abbreviate feature names to fit 15 character limit
df = pd.DataFrame({
    'features': features,
    'importance': importance
})

# Abbreviate feature names to fit the 15 character limit
abbreviated_features = [
    "Surf_Defect_Cnt",
    "Pattern_Align", 
    "Resistance",
    "Surf_Rough",
    "Temperature",
    "Voltage",
    "Wafer_Thick",
    "Chem_Conc",
    "Process_Time"
]

df['abbreviated_features'] = abbreviated_features

# Create horizontal bar chart with cliponaxis applied to the trace
fig = go.Figure(go.Bar(
    x=df['importance'],
    y=df['abbreviated_features'],
    orientation='h',
    marker_color='#1FB8CD',
    cliponaxis=False,
    hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
))

# Update layout
fig.update_layout(
    title="RF Feature Importance",
    xaxis_title="Importance",
    yaxis_title="Features",
    yaxis={'categoryorder': 'array', 'categoryarray': df['abbreviated_features'][::-1]}  # Reverse order for highest to lowest
)

# Save the chart
fig.write_image("feature_importance_chart.png")