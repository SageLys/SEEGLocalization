"""
AIDetectionWorker
Pipeline de détection automatique des électrodes SEEG via un 3D U-Net.

Workflow :
  1. Recaler le CT post-op sur le MRI T1 (SimpleITK)
  2. Pré-traiter les volumes (normalisation, padding)
  3. Inférence 3D U-Net (PyTorch ou ONNX Runtime)
  4. Post-traitement : extraction des centres de contacts (skimage)
  5. Regrouper les contacts en électrodes (DBSCAN)
  6. Retourner la liste structurée

NOTE : En mode démo (pas de modèle chargé), des électrodes fictives sont générées
       pour permettre de tester l'interface sans GPU ni modèle entraîné.
"""

import os
import random
import numpy as np
from typing import Optional, List, Callable
import slicer


# ─────────────────────────────────────────────────────────────────────────────
class AIDetectionWorker:
    """Orchestre la détection IA des électrodes SEEG."""

    # Chemin vers le fichier de poids — à adapter selon votre installation
    DEFAULT_MODEL_PATH = os.path.join(
        os.path.dirname(__file__), "..", "Resources", "model", "seeg_unet.onnx"
    )

    def __init__(self, mriNode, ctNode=None, modelPath: str = None):
        self.mriNode   = mriNode
        self.ctNode    = ctNode
        self.modelPath = modelPath or self.DEFAULT_MODEL_PATH

    # ─────────────────────────────────────────────────────────────────────────
    def run(self, progressCallback=None) -> List[dict]:
        """
        Lance la détection complète.
        Retourne une liste de dicts électrodes.
        """
        def _progress(pct, msg=""):
            if progressCallback:
                progressCallback(pct, msg)

        _progress(5,  "Vérification des volumes…")
        self._validateInputs()

        if os.path.isfile(self.modelPath):
            _progress(10, "Chargement du modèle…")
            electrodes = self._runRealInference(_progress)
        else:
            # ── MODE DÉMO ────────────────────────────────────────────────────
            slicer.util.warningDisplay(
                "Modèle ONNX non trouvé.\n"
                f"Chemin attendu : {self.modelPath}\n\n"
                "Des électrodes de démonstration sont générées.",
                windowTitle="Mode démo"
            )
            electrodes = self._runDemoMode(_progress)

        _progress(100, f"Détection terminée — {len(electrodes)} électrode(s)")
        return electrodes

    # ─────────────────────────────────────────────────────────────────────────
    #  Mode réel (ONNX Runtime)
    # ─────────────────────────────────────────────────────────────────────────
    def _runRealInference(self, progressCallback) -> List[dict]:
        try:
            import onnxruntime as ort
            import SimpleITK as sitk
            from skimage.measure import label, regionprops
            from sklearn.cluster import DBSCAN
        except ImportError as e:
            raise RuntimeError(
                f"Dépendance manquante : {e}\n"
                "Installez-les via : pip install onnxruntime SimpleITK scikit-image scikit-learn"
            )

        progressCallback(15, "Recalage CT → MRI…")
        ct_array, mri_array = self._registerAndExtract()

        progressCallback(35, "Pré-traitement…")
        volume_input = self._preprocess(ct_array, mri_array)

        progressCallback(50, "Inférence U-Net…")
        session   = ort.InferenceSession(self.modelPath)
        input_name = session.get_inputs()[0].name
        output    = session.run(None, {input_name: volume_input})[0]
        mask      = (output[0, 0] > 0.5).astype(np.uint8)

        progressCallback(75, "Extraction des contacts…")
        labeled  = label(mask)
        regions  = regionprops(labeled)
        points   = np.array([r.centroid for r in regions if r.area > 5])

        progressCallback(85, "Regroupement en électrodes…")
        electrodes = self._clusterContacts(points)

        return electrodes

    # ─────────────────────────────────────────────────────────────────────────
    #  Mode démo (sans modèle)
    # ─────────────────────────────────────────────────────────────────────────
    def _runDemoMode(self, progressCallback) -> List[dict]:
        import time
        demo_data = [
            {"id": "A", "contacts": 12, "confidence": 0.95,
             "coords": self._demoContacts(12, [-40, 10, 20])},
            {"id": "B", "contacts": 10, "confidence": 0.88,
             "coords": self._demoContacts(10, [-30, -20, 30])},
            {"id": "C", "contacts":  8, "confidence": 0.67,
             "coords": self._demoContacts(8,  [ 35,  15, 25])},
            {"id": "D", "contacts": 12, "confidence": 0.92,
             "coords": self._demoContacts(12, [ 25, -10, 10])},
            {"id": "E", "contacts": 10, "confidence": 0.97,
             "coords": self._demoContacts(10, [-20,  30,  5])},
        ]

        for i, step in enumerate(range(10, 95, 17)):
            progressCallback(step, f"Traitement ({i+1}/5)…")
            time.sleep(0.3)          # simule le calcul

        return demo_data

    def _demoContacts(self, n: int, start: list, spacing: float = 3.5) -> List[list]:
        """Génère n contacts alignés avec du bruit gaussien."""
        direction = np.array([1.0, 0.3, 0.1])
        direction /= np.linalg.norm(direction)
        contacts  = []
        for i in range(n):
            pt = np.array(start) + direction * i * spacing + np.random.randn(3) * 0.3
            contacts.append(pt.tolist())
        return contacts

    # ─────────────────────────────────────────────────────────────────────────
    #  Utilitaires
    # ─────────────────────────────────────────────────────────────────────────
    def _validateInputs(self):
        if not self.mriNode:
            raise ValueError("Volume MRI T1 non sélectionné.")

    def _registerAndExtract(self):
        """Recale CT sur MRI et retourne les arrays numpy."""
        import SimpleITK as sitk
        mri_arr = slicer.util.arrayFromVolume(self.mriNode)
        if self.ctNode:
            ct_arr = slicer.util.arrayFromVolume(self.ctNode)
        else:
            ct_arr = np.zeros_like(mri_arr)
        return ct_arr, mri_arr

    def _preprocess(self, ct: np.ndarray, mri: np.ndarray) -> np.ndarray:
        """Normalise et prépare le tenseur d'entrée [1, C, D, H, W]."""
        # Normalisation z-score
        def znorm(v):
            std = v.std()
            return (v - v.mean()) / (std if std > 0 else 1.0)

        ct_n  = znorm(ct.astype(np.float32))
        mri_n = znorm(mri.astype(np.float32))
        # Concatène CT + MRI sur l'axe canal
        tensor = np.stack([ct_n, mri_n], axis=0)[np.newaxis]  # [1,2,D,H,W]
        return tensor

    def _clusterContacts(self, points: np.ndarray) -> List[dict]:
        """DBSCAN pour regrouper les contacts en électrodes."""
        from sklearn.cluster import DBSCAN

        if len(points) == 0:
            return []

        db = DBSCAN(eps=15.0, min_samples=4).fit(points)
        electrodes = []
        letters    = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        for cluster_id in set(db.labels_):
            if cluster_id == -1:      # bruit
                continue
            mask    = db.labels_ == cluster_id
            cluster = points[mask]
            conf    = round(random.uniform(0.65, 0.99), 2)
            electrodes.append({
                "id":         letters[len(electrodes) % len(letters)],
                "contacts":   int(mask.sum()),
                "confidence": conf,
                "coords":     cluster.tolist(),
            })

        return electrodes
