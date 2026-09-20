# AgroAI Dataset Documentation

## 1. Overview
The field dataset (`backend/data/field_data.csv`) contains agronomic and environmental parameters collected per field sector.

## 2. Field Schema Columns

| Column | Unit / Range | Description |
| :--- | :--- | :--- |
| `field_id` | String | Unique identifier (e.g. `Field_A`, `Field_B`) |
| `crop` | String | Type of crop planted (e.g. `Rice`, `Tomato`, `Maize`, `Potato`) |
| `soil_moisture` | Percentage (%) | Amount of moisture available in soil |
| `soil_ph` | pH scale (0–14) | Soil acidity or alkalinity value |
| `temperature` | °C | Field ambient temperature |
| `humidity` | Percentage (%) | Relative air humidity |
| `rainfall` | mm | Measured precipitation |
| `water_availability` | Liters (L) | Available irrigation water in main reservoir |

## 3. Algorithm Data Consumers
- **K-Means Clustering**: Uses `soil_moisture`, `soil_ph`, `temperature`, and `humidity` to group fields into soil health zones.
- **Decision Tree Classifier**: Uses `soil_moisture` and `rainfall` to classify irrigation urgency.
- **CSP & AC-3**: Uses `field_id`, `water_availability`, and crop priority to resolve non-conflicting irrigation time slot schedules.

## 4. Demo Data vs Real Sources
- **Week 1**: All values represent curated **Demo Data** created for university laboratory prototyping.
- **Future Integration**: Data will be ingested from real IoT soil sensors and weather API feeds.
