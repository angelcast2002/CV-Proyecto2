# Proyecto de Detección de Cuerpo y Manos con MediaPipe

Este proyecto utiliza la biblioteca **MediaPipe** para realizar la detección de poses corporales y manos en imágenes, videos o transmisiones en vivo desde una cámara. Incluye dos scripts principales: uno para la detección de poses corporales y otro para la detección de manos.

## Contenido

- [Requisitos](#requisitos)
- [Uso](#uso)
  - [Detección de Cuerpo](#detección-de-cuerpo)
  - [Detección de Manos](#detección-de-manos)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Créditos](#créditos)

---

## Requisitos

- Python 3.8 o superior
- Bibliotecas necesarias:
  - `mediapipe`
  - `opencv-python`
  - `numpy`

---

## Uso
### Detección de Cuerpo
El script bodyDetection.py permite detectar poses corporales en imágenes, videos o transmisiones en vivo desde la cámara.

1. **Ejecuta el script:**
```
python bodyDetection.py
```
2. **Selecciona la fuente de entrada:**

- Cámara en vivo.
- Archivo de video (debes proporcionar la ruta del archivo).
- Imagen estática (debes proporcionar la ruta de la imagen).

3. **Resultados:**
- Si seleccionas una imagen, se mostrarán y guardarán las imágenes procesadas con las poses detectadas.
- Si seleccionas un video o la cámara, se mostrarán las poses detectadas en tiempo real.

### Detección de Manos
El script `handDetectorMediapipeCV.py` permite detectar manos en transmisiones en vivo desde la cámara.

**Resultados:**
Se mostrará la transmisión en vivo con las manos detectadas y sus puntos clave dibujados.

## Estructura del proyecto
CV-Proyecto2/
│
├── [bodyDetection.py](http://_vscodecontentref_/0)          # Script para detección de poses corporales
├── [handDetectorMediapipeCV.py](http://_vscodecontentref_/1) # Script para detección de manos
├── [pose_landmarker_heavy.task](http://_vscodecontentref_/2) # Modelo de MediaPipe para detección de cuerpo
├── [hand_landmarker.task](http://_vscodecontentref_/3)       # Modelo de MediaPipe para detección de manos
├── data/
│   └── output/               # Carpeta para guardar los resultados procesados
├── [README.md](http://_vscodecontentref_/4)                 # Documentación del proyecto
└── requirements.txt          # Dependencias del proyecto

## Créditos
Este proyecto utiliza la biblioteca MediaPipe para la detección de poses y manos. MediaPipe es una solución de visión por computadora desarrollada por Google.

Desarrollado por: **Diego Morales, Angel Castellanos y Alejandro Azurdia** para el curso "Vision por Computadoras" de la Universidad del Valle de Guatemala, en abril de 2025. 