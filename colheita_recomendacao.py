import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Recomendador de Condições de Cultivo", page_icon="🌾", layout="wide")
df = pd.read_csv(r'Crop_recommendation.csv')

# Recommendation system class
class CropRecommendationSystem:
    def __init__(self, df):
        self.df = df
        self.features = ['N', 'P', 'K', 'temperatura', 'umidade', 'pH', 'precipitação']
    
    def get_ideal_conditions(self, crop_label):
        """
        Retorna condições ideais (média, mínimo, máximo) e número de amostras para uma cultura específica
        """
        crop_data = self.df[self.df['label'] == crop_label]
        
        if crop_data.empty:
            return None, None, None, 0, f"Nenhum dado encontrado para a cultura: {crop_label}"
        
        # Calculate average, min, and max conditions
        avg_conditions = crop_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].mean().to_dict()
        min_conditions = crop_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].min().to_dict()
        max_conditions = crop_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].max().to_dict()
        num_samples = len(crop_data)
        
        # Map to Portuguese feature names
        avg_conditions_pt = {self.features[i]: avg_conditions[key] 
                            for i, key in enumerate(['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])}
        min_conditions_pt = {self.features[i]: min_conditions[key] 
                            for i, key in enumerate(['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])}
        max_conditions_pt = {self.features[i]: max_conditions[key] 
                            for i, key in enumerate(['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])}
        
        return avg_conditions_pt, min_conditions_pt, max_conditions_pt, num_samples, None

# Custom CSS for enhanced visuals
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s ease, transform 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .sidebar .sidebar-content {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 8px;
    }
    h1, h2, h3 {
        color: #2e7d32;
        font-family: 'Arial', sans-serif;
        margin-bottom: 10px;
    }
    /* Dropdown styling */
    .stSelectbox > div > div {
        background-color: #e8f5e9 !important; /* Match sidebar background */
        border: 1px solid #4CAF50 !important;
        border-radius: 1px;
        padding: 0px !important;
        color: #333333 !important; /* Dark text for contrast */
        min-height: 10px !important; /* Prevent truncation */
        line-height: 1.5 !important;
    }
    .stSelectbox > div > div > div {
        color: #333333 !important; /* Ensure dropdown options are dark */
        background-color: #ffffff !important; /* White background for dropdown options */
    }
    .stSelectbox > div > div > div:hover {
        background-color: #d4edda !important; /* Light green hover effect */
    }
    /* Table styling */
    .stTable {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-top: 10px;
    }
    .stTable table {
        color: #333333 !important;
        background-color: #ffffff !important;
        border-collapse: collapse;
    }
    .stTable th, .stTable td {
        border: 1px solid #dddddd !important;
        padding: 10px !important;
        color: #333333 !important;
        text-align: left;
    }
    .stTable th {
        background-color: #f5f5f5 !important;
        font-weight: bold;
    }
    .stTable tr:nth-child(even) {
        background-color: #fafafa !important;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app
def main():
    
    
    # Title and description in main area
    st.title("Recomendador de Condições de Cultivo")
    st.markdown("Selecione uma cultura na barra lateral para obter as condições ideais de cultivo com base em dados históricos.")
    
    # Initialize recommendation system
    recommender = CropRecommendationSystem(df)
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Entrada de Dados")
        crop_labels = df['label'].unique().tolist()
        crop_label = st.selectbox("Selecione a Cultura", crop_labels, index=0, help="Escolha uma cultura da lista")
        
        # Button to trigger recommendation
        submit = st.button("Obter Condições Ideais")
    
    # Main content area for results
    if submit:
        avg_conditions, min_conditions, max_conditions, num_samples, error = recommender.get_ideal_conditions(crop_label)
        
        if error:
            st.error(error)
        else:
            # Success message and sample count
            st.success(f"Condições Ideais para {crop_label}")
            st.write(f"Baseado em {num_samples} amostras")
            
            # Create DataFrame for display
            conditions_df = pd.DataFrame({
                'Parâmetro': list(avg_conditions.keys()),
                'Média': [avg_conditions[key] for key in avg_conditions],
                'Mínimo': [min_conditions[key] for key in avg_conditions],
                'Máximo': [max_conditions[key] for key in avg_conditions]
            })
            conditions_df[['Média', 'Mínimo', 'Máximo']] = conditions_df[['Média', 'Mínimo', 'Máximo']].round(2)
            
            # Display table
            st.subheader("Tabela de Condições")
            st.table(conditions_df)
            
            # Download button
            csv = conditions_df.to_csv(index=False)
            st.download_button(
                label="Baixar Condições (CSV)",
                data=csv,
                file_name=f"condicoes_{crop_label}.csv",
                mime="text/csv"
            )
            
            # Interactive Plotly bar chart
            st.subheader("Visualização em Barras")
            fig_bar = px.bar(
                conditions_df,
                x='Parâmetro',
                y='Média',
                labels={'Parâmetro': 'Parâmetro', 'Média': 'Valor Médio'},
                title=f"Condições Ideais para {crop_label} (Média)",
                color='Parâmetro',
                color_discrete_sequence=px.colors.qualitative.Set2,
                height=400
            )
            fig_bar.update_layout(
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                margin=dict(l=20, r=20, t=50, b=20),
                title_x=0.5
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Interactive Plotly radar chart
            st.subheader("Visualização em Radar")
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=conditions_df['Média'],
                theta=conditions_df['Parâmetro'],
                fill='toself',
                name='Média',
                line=dict(color='#4CAF50')
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=conditions_df['Mínimo'],
                theta=conditions_df['Parâmetro'],
                fill='toself',
                name='Mínimo',
                line=dict(color='#FF9800', dash='dash')
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=conditions_df['Máximo'],
                theta=conditions_df['Parâmetro'],
                fill='toself',
                name='Máximo',
                line=dict(color='#2196F3', dash='dash')
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True),
                    bgcolor='rgba(0,0,0,0)'
                ),
                showlegend=True,
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                margin=dict(l=20, r=20, t=50, b=20),
                title=f"Condições para {crop_label} (Média, Mínimo, Máximo)",
                title_x=0.5
            )
            st.plotly_chart(fig_radar, use_container_width=True)

if __name__ == "__main__":
    main()