# EECS-298-

Creating a weather prediction system using an embedded device is a great project that combines hardware and software skills. Here's an overview of how you can approach it:

🔧 1. Hardware Requirements
Microcontroller or SBC (Single Board Computer):

Examples: Raspberry Pi, ESP32, Arduino + Wi-Fi module.

Sensors (for local weather data):

Temperature/Humidity: DHT11, DHT22, or BME280.

Pressure: BMP180 or BME280.

Rain sensor, Wind speed/direction (optional for more detail).

Connectivity: Wi-Fi module (built-in in ESP32/Raspberry Pi) to fetch online weather data or upload data to a server.

🧠 2. Software Components
Option A: On-Device Weather Prediction (Simple Models)
Use basic models (e.g., linear regression) to predict temperature/humidity trends.

Train models off-device (e.g., Python + scikit-learn), convert coefficients to C/C++ code, and run on the embedded system.

Option B: Cloud-Assisted Prediction
Device collects sensor data → sends to a cloud service or local server running advanced models (like LSTM/RNN).

Server returns predictions.

Suitable for more powerful forecasts but needs internet and a server backend.

Option C: Fetch Weather Predictions
Device directly accesses weather APIs (like OpenWeatherMap) to show near-term forecasts.


# Report 
For system building projects:

Introduction

Background

Problem Statement

Details of the System

Main Results

Future Work

Conclusions
