# Dockerfile pour bâtir l'image Docker de l'environnement d'exécution Cloud Run
FROM prefecthq/prefect:3.4.25-python3.14

# Install Python requirements
COPY requirements.txt ./requirements.txt
RUN python -m pip install -r requirements.txt

# Create workspace directory
RUN mkdir -p /workspace

# Copy project files
COPY ./flows /workspace/flows