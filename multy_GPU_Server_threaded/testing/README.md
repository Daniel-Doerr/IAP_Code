# Test Setup für main.py

Dieses Verzeichnis enthält alle Dateien zum Testen der `main.py` mit einem lokalen Webserver.

## Dateien

- `test_server.py` - Lokaler Webserver der Jobs bereitstellt
- `start_test_server.bat` - Startet den Server automatisch
- `requirements.txt` - Benötigte Python-Pakete
- `test_images/` - Verzeichnis für Testbilder
- `generated_results/` - Hier werden generierte Bilder gespeichert

## Setup und Start

### 1. Abhängigkeiten installieren
```bash
pip install -r requirements.txt
```

### 2. Testbilder hinzufügen
Lege beliebige Bilder (.jpg, .png, etc.) **direkt in das `testing/` Verzeichnis**. Die Server suchen automatisch nach Bildern im gleichen Verzeichnis und laden sie von dort.

### 3. Test Server starten
```bash
# Linux
chmod +x start_test_server.sh
./start_test_server.sh

# Windows
start_test_server.bat

# Oder direkt mit Python
python test_server.py
```

Der Server läuft dann auf: http://localhost:8000

### 4. main.py testen
In einem neuen Terminal:

**Option A: Einfacher Test (ohne ComfyUI/GPU-Abhängigkeiten)**
```bash
# Im testing/ Verzeichnis
python test_main.py
```

**Option B: Echte main.py (benötigt ComfyUI und GPU-Setup)**
```bash
# Im Hauptverzeichnis 
cd ..
python main.py
```

## Funktionsweise

### Test Server
- Stellt Jobs mit verschiedenen Workflows bereit (FLUX_Kontext, IP_Adapter_SDXL)
- Wechselt automatisch zwischen Workflows nach 3 Jobs
- Simuliert verschiedene Patienten und Tiere
- Empfängt und speichert generierte Bilder im `generated_results/` Ordner

### Server Endpunkte
- `POST /token` - Authentifizierung (Password: "Password")
- `GET /job` - Holt einen neuen Job mit Bild und Metadaten
- `POST /job` - Sendet generierte Ergebnisse zurück
- `GET /status` - Zeigt aktuellen Server-Status

### Workflow-Rotation
Der Server wechselt automatisch zwischen den Workflows:
1. 3 Jobs mit FLUX_Kontext
2. 3 Jobs mit IP_Adapter_SDXL  
3. Wieder zu FLUX_Kontext, usw.

### Testdaten
- 5 verschiedene Patienten mit unterschiedlichen Tieren
- Rotiert durch verfügbare Testbilder
- Verschiedene Tiertypen: bear, cat, dog, rabbit, other

## Überwachung

### Server Status
Besuche http://localhost:8000/status um den aktuellen Status zu sehen.

### Logs
Der Server zeigt alle verarbeiteten Jobs in der Konsole an:
```
Sending job job_0001: FLUX_Kontext - Teddy (bear)
Received and saved result for job_0001: job_0001_20240709_143022.png
```

### Generierte Bilder
Alle generierten Bilder werden in `generated_results/` gespeichert mit dem Format:
`{job_id}_{timestamp}.png`

## Stoppen
- Server: Ctrl+C in der Server-Konsole
- main.py: Ctrl+C in der main.py-Konsole
