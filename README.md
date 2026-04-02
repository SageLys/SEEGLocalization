# SEEG Localization Module (G05) — Guide d'installation

## Architecture du projet

```
SEEGLocalization/
│
├── SEEGLocalization.py              ← Module principal (IHM + logique Slicer)
├── CMakeLists.txt                   ← Packaging optionnel en extension Slicer
├── README.md
│
├── SEEGLocalizationLib/
│   ├── __init__.py
│   ├── electrode_manager.py         ← Gestion MRML des électrodes (CRUD, couleurs, filtres)
│   ├── ai_detection.py              ← Pipeline IA : recalage → U-Net → clustering
│   └── export_manager.py            ← Export CSV / JSON / BIDS / MRML
│
└── Resources/
    ├── Icons/                       ← Icône du module (optionnel)
    └── model/
        └── seeg_unet.onnx           ← Poids du modèle (à placer ici)
```

---

## Prérequis

| Logiciel | Version minimale |
|---|---|
| 3D Slicer | 5.4+ |
| Python (inclus dans Slicer) | 3.9+ |

---

## Installation — méthode rapide (sans compilation)

### 1. Télécharger / cloner le module

```bash
git clone https://github.com/your-org/SEEGLocalization.git
# ou télécharger le ZIP et extraire
```

### 2. Activer le mode développeur dans Slicer

```
Edit > Application Settings > Developer > ✅ Enable developer mode
```
Redémarrer Slicer.

### 3. Ajouter le chemin du module

```
Edit > Application Settings > Modules > Additional module paths
```
Cliquer sur **+** et sélectionner le dossier `SEEGLocalization/` (celui qui contient `SEEGLocalization.py`).

Redémarrer Slicer.

### 4. Ouvrir le module

Dans la barre de recherche de modules (loupe en haut) :
```
SEEG Localization Module
```
→ Le panneau G05 apparaît à gauche.

---

## Installation des dépendances Python

Ouvrir la console Python de Slicer : **View > Python Interactor**, puis :

```python
import subprocess, sys
packages = [
    "onnxruntime",          # inference ONNX U-Net
    "SimpleITK",            # recalage CT/MRI
    "scikit-image",         # post-traitement masques
    "scikit-learn",         # DBSCAN clustering
]
for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
print("✔ Installation terminée")
```

---

## Placer le modèle IA

Copier votre fichier `seeg_unet.onnx` dans :

```
SEEGLocalization/Resources/model/seeg_unet.onnx
```

> **Sans modèle :** le module fonctionne en **mode démo** — 5 électrodes fictives
> sont générées automatiquement pour tester l'interface.

---

## Guide d'utilisation

### Onglet Data
1. Charger un volume MRI T1 et un CT post-op via les sélecteurs **ou** le bouton "Charger depuis le disque"
2. Cliquer sur **Run AI Detection (3D U-Net)**
3. La barre de progression indique l'avancement
4. Les électrodes apparaissent dans la vue 3D et dans la liste

### Onglet Electrode
- **Transparency** : ajuste l'opacité du cerveau et des électrodes en temps réel
- **Labels** : affiche/masque les noms et numéros de contacts
- **Electrodes List** : cliquer sur une ligne → l'électrode se met en surbrillance (blanche),
  les vues 2D (Axial/Sagittal/Coronal) se centrent sur le premier contact,
  les coordonnées X Y Z s'affichent sous le tableau
- **Seuil de confiance** : masque les électrodes sous le seuil
- **Show by status** : filtre Validée / Confiance moyenne / Faible confiance

### Onglet Export
| Format | Fichier généré | Usage |
|---|---|---|
| CSV | `Patient_001_SEEG_electrodes.csv` | Excel, tableur |
| JSON | `Patient_001_SEEG_electrodes.json` | API, pipeline custom |
| BIDS | `sub-Patient001_electrodes.tsv` + `_coordsystem.json` | Publication, recherche |
| MRML | `Patient_001_SEEG_scene.mrb` | Partage scène Slicer |

---

## Intégrer votre propre modèle U-Net

Le fichier `SEEGLocalizationLib/ai_detection.py` est le seul à modifier.

**Exporter en ONNX depuis PyTorch :**

```python
import torch

model = MySEEGUNet()
model.load_state_dict(torch.load("seeg_unet.pth"))
model.eval()

dummy = torch.zeros(1, 2, 128, 128, 128)   # [batch, channels(CT+MRI), D, H, W]
torch.onnx.export(
    model, dummy, "seeg_unet.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)
```

Copier `seeg_unet.onnx` dans `Resources/model/`.

---

## Architecture technique

```
IHM (SEEGLocalizationWidget)
│
├── [Data Tab]
│   ├── qMRMLNodeComboBox  → sélection volumes
│   ├── QPushButton        → déclenche AIDetectionWorker.run()
│   └── QProgressBar       → feedback visuel
│
├── [Electrode Tab]
│   ├── ctkSliderWidget    → ElectrodeManager.setGlobalOpacity()
│   ├── QCheckBox          → ElectrodeManager.setLabelsVisible()
│   ├── QTableWidget       → ElectrodeManager.highlightElectrode()
│   │                         + SEEGLocalizationLogic.centerViewsOnPoint()
│   ├── ctkSliderWidget    → ElectrodeManager.applyConfidenceThreshold()
│   └── QCheckBox ×3       → ElectrodeManager.applyStatusFilter()
│
└── [Export Tab]
    ├── QRadioButton ×4    → choix format
    └── QPushButton        → ExportManager.export()

SEEGLocalizationLogic
├── runAIDetection()       → AIDetectionWorker
└── centerViewsOnPoint()   → vtkMRMLSliceNode.JumpSliceByCentering()

AIDetectionWorker
├── _registerAndExtract()  → SimpleITK
├── _preprocess()          → numpy
├── _runRealInference()    → onnxruntime + skimage + sklearn
└── _runDemoMode()         → données fictives

ElectrodeManager
└── vtkMRMLMarkupsFiducialNode × N électrodes

ExportManager
└── csv / json / bids / slicer.util.saveScene()
```

---

## FAQ

**Q : Le module n'apparaît pas dans Slicer.**
→ Vérifier que le chemin dans `Additional module paths` pointe vers le dossier contenant `SEEGLocalization.py`, pas un dossier parent.

**Q : Erreur `ModuleNotFoundError: onnxruntime`**
→ Réexécuter le bloc d'installation pip dans la console Python de Slicer.

**Q : Les électrodes ne s'affichent pas en 3D.**
→ S'assurer qu'un modèle de cerveau est chargé (via `File > Add Data`) et que l'opacité Brain n'est pas à 0.

**Q : Comment tester sans données patient ?**
→ Utiliser les données de démonstration Slicer : `File > Download Sample Data > MRHead`.
Le mode démo génère des électrodes fictives sans CT.
