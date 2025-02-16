# pyrat2

PyRAT2: Python Room Acoustics Toolkit
(Optional: Add an image or logo here)

A Python toolkit for analyzing, processing, and visualizing room acoustics measurements.
📌 Overview
PyRAT2 (Python Room Acoustics Toolkit) is a library designed for the analysis and visualization of room impulse responses (RIRs) and acoustic parameters. It provides tools for reverberation time estimation, energy decay analysis, spatial acoustics comparisons, and more.

This toolkit is useful for acoustic researchers, audio engineers, and archaeoacoustics specialists working with room acoustics data.

🚀 Features
✔ Impulse Response Processing: Load, visualize, and manipulate RIRs
✔ Acoustic Parameter Estimation: Compute T30, T20, T10, EDT, C50, C80, D50
✔ Spatial Acoustics Analysis: Compare responses from different microphone positions
✔ Filtering & Signal Processing: Octave-band and third-octave analysis
✔ Customizable Analysis Pipelines: Modular functions for research workflows
✔ Support for Auralization: Prepares RIR data for immersive sound simulations

📦 Installation
You can install PyRAT2 via pip (if published) or from source:

bash
Copier
Modifier
pip install pyrat2
or

bash
Copier
Modifier
git clone https://github.com/lunavalentin/pyrat2.git
cd pyrat2
pip install -r requirements.txt
📝 Usage
Here’s a quick example of how to load an impulse response and compute its reverberation time:

python
Copier
Modifier
import pyrat2 as pr

# Load an impulse response
rir = pr.load_rir("example_rir.wav")

# Compute reverberation time
t30 = pr.compute_reverberation_time(rir, method="T30")
print(f"Reverberation Time (T30): {t30} seconds")

# Visualize the impulse response
pr.plot_rir(rir)
For more examples, check the documentation.

📂 Project Structure
bash
Copier
Modifier
pyrat2/
│── pyrat2/                 # Main library code
│   ├── __init__.py         # Package initialization
│   ├── processing.py       # Signal processing functions
│   ├── analysis.py         # Acoustic parameter calculations
│   ├── visualization.py    # Plotting and visualization
│── examples/               # Example scripts
│── tests/                  # Unit tests
│── README.md               # Project documentation
│── setup.py                # Installation script
│── requirements.txt        # Dependencies
📖 Documentation
For detailed usage instructions, visit the Wiki.

🤝 Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (feature-branch).
Commit your changes and push to GitHub.
Open a pull request.
🔗 License
This project is licensed under the MIT License. See LICENSE for details.

🛠️ Acknowledgments
Developed as part of room acoustics research at Stanford CCRMA.
Developped in contribution with Sara MArtin and Peter Svensson (NTNU), & Jonathan Able (CCRMA)
Inspired by various open-source acoustics toolkits.
