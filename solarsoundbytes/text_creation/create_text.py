import pandas as pd
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
# api_key = os.getenv("API_KEY")

api_key = os.getenv("API_KEY")

openai.api_key = api_key

def create_text_from_sent_analy_df(data_twitter, data_news, data_1):

    prompt = f"""
    Hier sind monatliche Sentimentdaten zum Thema 'Klimawandel' aus Nachrichtenartikeln und Twitter:
    Bei Betrachtung der folgenden Dataframes
    {data_twitter}
    {data_news}
    {data_1}

    Bitte fasse die Entwicklung der öffentlichen Meinung zusammen..
    - Erkläre, ob die Wahrnehmung in sozialen Medien und in Nachrichtenmedien unterschiedlich war.
    - Wichtige Ereignisse, in diesem Zeitraum: Juli 2024 – Wärmstes Jahr: Globaltemperatur erstmals >1,5 °C über vorindustriellem Niveau.

    April 2023 – Hitzewelle Asien: Rekordtemperaturen in Indien, Bangladesch, China, Thailand, Vietnam.

    September 2023 – Libyen-Flut: Dammbruch nach Sturmtief „Daniel“, >11.000 Tote.

    Juni–August 2022 – Pakistan-Flut: 1.700 Tote, 1/3 des Landes unter Wasser.

    Mai–September 2023 – Waldbrände Kanada: Größte je gemessene Fläche verbrannt (173.000 km²).

    Februar–März 2023 – Zyklon Freddy: >1.190 Tote in Südostafrika.

    Oktober 2024 – Spanien-Flut: Extremregen, >230 Tote.

    Februar–November 2023 – Antarktis-Meereisminimum: Rekordniedrige Eisausdehnung.

    Mai 2024 – Gletscherverlust Venezuela: Letzter Gletscher verschwindet.

    2024 (Jahr) – Amazonas-Dürre & Abholzung: 88 Mio. ha Regenwald zerstört, Rekord-Trockenheit.
    Dezember 2023 – COP28 (Dubai, VAE)
    → Erstes globales Bekenntnis zum „Übergang weg von fossilen Brennstoffen“
    Ein historischer Schritt, wenn auch ohne klare Verpflichtungen zum vollständigen Ausstieg.

    August 2022 – USA: „Inflation Reduction Act“
    → 369 Mrd. USD für Klima und saubere Energie
    Größtes Klimapaket der US-Geschichte; stärkt Wind, Solar, Speicher, E-Autos.

    Januar 2023 – EU: „REPowerEU“-Maßnahmen
    → Beschleunigung des Ausbaus erneuerbarer Energien
    Ziel: Unabhängigkeit von russischer Energie und Förderung grüner Technologien.

    Oktober 2023 – Deutschland: Solarpaket I beschlossen
    → Fördervereinfachung für Solaranlagen und Balkonkraftwerke
    Stärkt bürgernahe Energiewende, steigert Akzeptanz erneuerbarer Energien.

    März 2024 – China: Investitionsplan für grüne Industrie
    → Mehr als 1 Billion Yuan (~140 Mrd. USD) für Solar, Wind, Speicher & Netze
    Signalisiert geopolitisches Engagement in globaler Energiewende.

    - Schreibe einen strukturierten Text mit etwa 100 Wörtern.
    auf englisch bitte
    """

    client = OpenAI(api_key = api_key)
    response = client.chat.completions.create(
        model="gpt-4o",  # oder "gpt-4o" für schnelleren Output
        messages=[
            {"role": "system", "content": "Du bist ein datenanalytischer Journalist."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1000
    )

    # 🖨️ Ausgabe
    # print(response.choices[0].message.content)
    return response.choices[0].message.content

# output = create_text_from_sent_analy_df(result_twitter, result_news)
# print(output)
