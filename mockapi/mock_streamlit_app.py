import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Sistema de Recomendação de Culturas", layout="wide")
st.title("🌾 Sistema de Recomendação de Culturas")
st.markdown("Clique no mapa ou insira uma localização para obter recomendações de culturas com base em condições ambientais simuladas.")

def fake_weather_api(lat, lon, season="wet"):
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return {"cod": 400, "message": "Latitude ou longitude inválida"}
    
    if abs(lat) <= 23.5:  
        if lon % 10 < 5:  
            temp_range = (25, 35)  
            humidity_range = (80, 95)  
            rainfall_hourly = (2, 10) if season == "wet" else (0, 3)
        else:  
            temp_range = (22, 32)
            humidity_range = (70, 90)
            rainfall_hourly = (1, 7) if season == "wet" else (0, 2)
    elif 23.5 < abs(lat) <= 50:  
        temp_range = (10, 25)
        humidity_range = (50, 80)
        rainfall_hourly = (0, 5) if season == "wet" else (0, 2)
    else:  
        temp_range = (0, 15)
        humidity_range = (40, 70)
        rainfall_hourly = (0, 3)
    
    error_chance = np.random.random()
    if error_chance < 0.03:
        return {"cod": 429, "message": "Limite de chamadas da API excedido"}
    elif error_chance < 0.06:
        return {"cod": 500, "message": "Erro interno do servidor"}
    
    temp = np.random.uniform(*temp_range)
    humidity = np.random.uniform(*humidity_range)
    rainfall = np.random.uniform(*rainfall_hourly)
    if not all(np.isfinite([temp, humidity, rainfall])):
        return {"cod": 500, "message": "Erro: Valores climáticos inválidos gerados"}
    
    return {
        "main": {
            "temp": float(temp),
            "humidity": float(humidity)
        },
        "rain": {
            "1h": float(rainfall)
        }
    }

def fake_soil_api(lat, lon):
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return {"error": "Latitude ou longitude inválida"}
    
    if abs(lat) <= 23.5:  
        if lon % 10 < 5: 
            ph_range = (4.5, 6.0)  
            n_range = (100, 140) 
            p_range = (50, 80) 
            k_range = (40, 70)  
        else:  
            ph_range = (5.5, 6.5)
            n_range = (80, 120)
            p_range = (40, 60)
            k_range = (30, 50)
    elif 23.5 < abs(lat) <= 50:  
        ph_range = (6.0, 7.5)
        n_range = (50, 100)
        p_range = (20, 50)
        k_range = (20, 40)
    else:  
        ph_range = (6.5, 8.0)
        n_range = (20, 60)
        p_range = (10, 30)
        k_range = (10, 25)
    
    if np.random.random() < 0.05:
        return {"error": "Erro na API SoilGrids: dados indisponíveis"}
    
    ph = np.random.uniform(*ph_range)
    nitrogen = np.random.uniform(*n_range)
    phosphorus = np.random.uniform(*p_range)
    potassium = np.random.uniform(*k_range)
    if not all(np.isfinite([ph, nitrogen, phosphorus, potassium])):
        return {"error": "Erro: Valores de solo inválidos gerados"}
    
    return {
        "properties": {
            "layers": [
                {"name": "phh2o", "ranges": [{"values": [{"M": float(ph)}]}]},
                {"name": "nitrogen", "ranges": [{"values": [{"M": float(nitrogen)}]}]},
                {"name": "phosphorus", "ranges": [{"values": [{"M": float(phosphorus)}]}]},
                {"name": "potassium", "ranges": [{"values": [{"M": float(potassium)}]}]}
            ]
        }
    }

@st.cache_data
def get_env_data(lat, lon, season="wet"):
    weather = fake_weather_api(lat, lon, season)
    if "cod" in weather and weather["cod"] != 200:
        raise ValueError(weather["message"])
    soil = fake_soil_api(lat, lon)
    if "error" in soil:
        raise ValueError(soil["error"])
    soil_layers = {layer["name"]: layer["ranges"][0]["values"][0]["M"] for layer in soil["properties"]["layers"]}
    data = {
        "temperature": weather["main"]["temp"],
        "humidity": weather["main"]["humidity"],
        "rainfall": weather["rain"]["1h"] * 24 * 30,  
        "ph": soil_layers["phh2o"],
        "N": soil_layers["nitrogen"],
        "P": soil_layers["phosphorus"],
        "K": soil_layers["potassium"]
    }
    
    for key, value in data.items():
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"Valor inválido para {key}: {value} (tipo: {type(value)})")
    return data

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Crop_recommendation.csv")  
        return df
    except FileNotFoundError:
        st.error("Erro: Arquivo 'Crop_recommendation.csv' não encontrado. Faça o download do dataset do Kaggle e coloque no diretório correto.")
        return None

@st.cache_resource
def train_ml_model(df):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    model.fit(X, y)
    return model

def plot_conditions(location_data, crop_ranges, selected_crop):
    fig, ax = plt.subplots(figsize=(10, 6))
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    location_values = [location_data[f] for f in features]
    crop_row = crop_ranges[crop_ranges['label'][0] == selected_crop]
    
    min_values = [crop_row[f]['min'].iloc[0] for f in features]
    max_values = [crop_row[f]['max'].iloc[0] for f in features]
    
    x = np.arange(len(features))
    ax.bar(x - 0.2, location_values, 0.2, label='Condições Atuais', color='blue')
    ax.bar(x, min_values, 0.2, label='Mínimo da Cultura', color='green', alpha=0.5)
    ax.bar(x + 0.2, max_values, 0.2, label='Máximo da Cultura', color='red', alpha=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Nitrogênio', 'Fósforo', 'Potássio', 'Temperatura', 'Umidade', 'pH', 'Precipitação'], rotation=45)
    ax.set_title(f'Condições vs. Requisitos de {selected_crop.capitalize()}')
    ax.legend()
    plt.tight_layout()
    return fig

def get_crop_ranges(df):
    return df.groupby('label').agg({
        'N': ['min', 'max'], 'P': ['min', 'max'], 'K': ['min', 'max'],
        'temperature': ['min', 'max'], 'humidity': ['min', 'max'],
        'ph': ['min', 'max'], 'rainfall': ['min', 'max']
    }).reset_index()

def main():
    df = load_data()
    if df is None:
        return
    crop_ranges = get_crop_ranges(df)  
    ml_model = train_ml_model(df)

    
    if 'lat' not in st.session_state:
        st.session_state.lat = -23.5505  
    if 'lon' not in st.session_state:
        st.session_state.lon = -46.6333  

    
    try:
        st.session_state.lat = float(st.session_state.lat)
    except (TypeError, ValueError):
        st.session_state.lat = -23.5505  
    try:
        st.session_state.lon = float(st.session_state.lon)
    except (TypeError, ValueError):
        st.session_state.lon = -46.6333  

    st.subheader("📍 Insira ou Clique no Mapa para Selecionar Localização")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.lat = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=st.session_state.lat,
            step=0.0001,
            key="lat_input",
            help="Ex.: -23.5505 para São Paulo. Clique no mapa para atualizar."
        )
    with col2:
        st.session_state.lon = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=st.session_state.lon,
            step=0.0001,
            key="lon_input",
            help="Ex.: -46.6333 para São Paulo. Clique no mapa para atualizar."
        )

    st.subheader("🗺️ Mapa Interativo")
    m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=10)
    folium.Marker(
        [st.session_state.lat, st.session_state.lon],
        popup=f"Local Selecionado<br>Latitude: {st.session_state.lat:.4f}<br>Longitude: {st.session_state.lon:.4f}",
        tooltip=f"Lat: {st.session_state.lat:.4f}, Lon: {st.session_state.lon:.4f}"
    ).add_to(m)
    map_data = st_folium(m, width=700, height=300, key="map")
    
    
    if map_data and map_data.get("last_clicked"):
        try:
            st.session_state.lat = float(map_data["last_clicked"]["lat"])
            st.session_state.lon = float(map_data["last_clicked"]["lng"])
        except (TypeError, ValueError):
            st.error("Erro: Coordenadas do mapa inválidas. Usando valores padrão.")
            st.session_state.lat = -23.5505
            st.session_state.lon = -46.6333
        st.rerun()  

    st.subheader("🌦️ Estação do Ano")
    season = st.selectbox("Selecione a estação", ["Chuvosa", "Seca"], help="Afeta a simulação de precipitação")
    season = "wet" if season == "Chuvosa" else "dry"

    st.subheader("🌍 Dados Ambientais (Opcional)")
    manual_input = st.checkbox("Inserir dados ambientais manualmente")
    if manual_input:
        st.write("Insira as condições ambientais para sua localização:")
        col3, col4 = st.columns(2)
        with col3:
            N = st.slider("Nitrogênio (N, mg/kg)", 0, 140, 80, help="Teor de nitrogênio no solo (0-140)")
            P = st.slider("Fósforo (P, mg/kg)", 0, 145, 40, help="Teor de fósforo no solo (5-145)")
            K = st.slider("Potássio (K, mg/kg)", 0, 205, 45, help="Teor de potássio no solo (5-205)")
            ph = st.slider("pH do Solo", 3.5, 10.0, 6.5, step=0.1, help="Acidez do solo (3.5-9.9)")
        with col4:
            temp = st.slider("Temperatura (°C)", 8.0, 44.0, 25.0, step=0.1, help="Temperatura média (8.8-43.7)")
            humidity = st.slider("Umidade (%)", 14, 100, 80, help="Umidade relativa do ar (14.3-99.9)")
            rainfall = st.slider("Precipitação (mm/mês)", 20, 300, 150, help="Chuva média mensal (20.2-298.7)")
        location_data = {
            'N': float(N), 'P': float(P), 'K': float(K), 'temperature': float(temp),
            'humidity': float(humidity), 'ph': float(ph), 'rainfall': float(rainfall)
        }
        
        for key, value in location_data.items():
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                st.error(f"Erro: Valor inválido para {key}: {value} (tipo: {type(value)}). Corrija os dados de entrada.")
                return
    else:
        with st.spinner("Obtendo dados ambientais simulados..."):
            try:
                location_data = get_env_data(st.session_state.lat, st.session_state.lon, season)
            except ValueError as e:
                st.error(f"Erro: {e}")
                return

    st.subheader("🌱 Condições da Localização")
    st.write(f"**Latitude**: {st.session_state.lat:.4f}, **Longitude**: {st.session_state.lon:.4f}")
    st.write(f"**Condições**: N={location_data['N']:.2f} mg/kg, P={location_data['P']:.2f} mg/kg, "
             f"K={location_data['K']:.2f} mg/kg, Temperatura={location_data['temperature']:.2f}°C, "
             f"Umidade={location_data['humidity']:.2f}%, pH={location_data['ph']:.2f}, "
             f"Precipitação={location_data['rainfall']:.2f} mm/mês")

    if st.button("Obter Recomendações de Culturas"):
        feature_order = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        try:
            location_df = pd.DataFrame([location_data])[feature_order]
            
            if location_df.isnull().values.any():
                st.error("Erro: Dados de entrada contêm valores nulos. Verifique os dados.")
                return
            if not all(location_df.dtypes == np.float64):
                st.error(f"Erro: Tipos inválidos em {location_df.dtypes[location_df.dtypes != np.float64].index.tolist()}. Todos os valores devem ser numéricos.")
                return
            
            with open("location_data_log.txt", "a") as f:
                f.write(f"location_data: {location_data}\n")
                f.write(f"location_df dtypes: {location_df.dtypes.to_dict()}\n")
            ml_prediction = ml_model.predict(location_df)
            st.subheader("🤖 Cultura Recomendada por Machine Learning")
            st.success(f"✅ {ml_prediction[0].capitalize()}")
            
            crop_images = {
                "rice": "https://via.placeholder.com/150?text=Arroz",
                "maize": "https://via.placeholder.com/150?text=Milho",
                "coffee": "https://via.placeholder.com/150?text=Café"
            }
            if ml_prediction[0] in crop_images:
                st.image(crop_images[ml_prediction[0]], caption=ml_prediction[0].capitalize(), width=150)
            st.pyplot(plot_conditions(location_data, crop_ranges, ml_prediction[0]))
        except ValueError as e:
            st.error(f"Erro na predição de ML: {str(e)}. Verifique se os dados estão dentro dos intervalos válidos.")
        except Exception as e:
            st.error(f"Erro inesperado na predição de ML: {str(e)}. Verifique os dados de entrada.")

if __name__ == "__main__":
    main()