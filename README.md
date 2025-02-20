# PyRAT2: Python Room Acoustics Toolkit

A Python toolkit for analyzing, processing, and visualizing room acoustics measurements.

## 📌 Overview
PyRAT2 (Python Room Acoustics Toolkit) is a library designed for the analysis and visualization of room impulse responses (RIRs) and acoustic parameters. It provides tools for reverberation time estimation, energy decay analysis, spatial acoustics comparisons, and more.

This toolkit is useful for acoustic researchers, audio engineers, and archaeoacoustics specialists working with room acoustics data.

## 🚀 Features
- **Impulse Response Processing**: Load, visualize, and manipulate RIRs  
- **Acoustic Parameter Estimation**: Compute **T30, T20, T10, EDT, C50, C80, D50**  
- **Spatial Acoustics Analysis**: Compare responses from different microphone positions  
- **Filtering & Signal Processing**: Octave-band and third-octave analysis  
- **Customizable Analysis Pipelines**: Modular functions for research workflows  
- **Support for Auralization**: Prepares RIR data for immersive sound simulations 

## 📦 Installation
You can install PyRAT2 via pip (if published) or from source:

bash
pip install pyrat2

or

git clone https://github.com/lunavalentin/pyrat2.git
cd pyrat2
pip install -r requirements.txt

## 🛠️ Acknowledgments
Developed as part of room acoustics research at Stanford CCRMA.
Developped in contribution with Sara MArtin and Peter Svensson (NTNU), & Jonathan Able (CCRMA)
Inspired by various open-source acoustics toolkits.
