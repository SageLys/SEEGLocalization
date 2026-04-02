"""
ElectrodeManager
Gère la création, la modification et la suppression des électrodes SEEG
sous forme de nœuds MRML Markups dans 3D Slicer.

Structure d'une électrode :
{
    "id":         "A",
    "contacts":   12,
    "confidence": 0.95,
    "status":     "validated",   # "validated" | "average" | "low"
    "visible":    True,
    "color":      [0.0, 1.0, 0.0],   # RGB 0-1
    "coords":     [[x1,y1,z1], [x2,y2,z2], ...]   # coordonnées RAS de chaque contact
}
"""

import slicer
import vtk
import numpy as np
from typing import Optional, List, Dict


# Couleurs par défaut pour les électrodes (cycle automatique)
DEFAULT_COLORS = [
    [0.0, 1.0, 0.0],    # vert
    [1.0, 0.5, 0.0],    # orange
    [0.2, 0.6, 1.0],    # bleu
    [1.0, 1.0, 0.0],    # jaune
    [0.8, 0.0, 0.8],    # violet
    [0.0, 1.0, 1.0],    # cyan
    [1.0, 0.2, 0.2],    # rouge clair
    [0.5, 1.0, 0.5],    # vert clair
]


class ElectrodeManager:
    """Gère l'ensemble des électrodes SEEG dans la scène MRML."""

    def __init__(self):
        self._electrodes: Dict[str, dict] = {}        # id → metadata
        self._nodes:      Dict[str, object] = {}      # id → vtkMRMLMarkupsFiducialNode
        self._highlighted: Optional[str] = None
        self._colorIndex  = 0
        self._confThreshold = 0.0
        self._statusFilter  = {"validated": True, "average": True, "low": True}

    # ─────────────────────────────────────────────────────────────────────────
    #  CRUD
    # ─────────────────────────────────────────────────────────────────────────
    def addElectrode(self, elec: dict):
        """
        Ajoute une électrode à la scène.
        elec doit contenir : id, contacts, confidence, coords (list of [x,y,z])
        """
        elec_id = elec["id"]

        # Status automatique selon confiance
        conf = elec.get("confidence", 0.0)
        if conf >= 0.90:
            status = "validated"
        elif conf >= 0.70:
            status = "average"
        else:
            status = "low"
        elec["status"] = elec.get("status", status)

        # Couleur automatique si non fournie
        if "color" not in elec:
            elec["color"] = DEFAULT_COLORS[self._colorIndex % len(DEFAULT_COLORS)]
            self._colorIndex += 1

        elec["visible"] = elec.get("visible", True)
        self._electrodes[elec_id] = elec

        # Crée le nœud MRML
        node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode", f"SEEG_{elec_id}")
        node.GetDisplayNode().SetSelectedColor(*elec["color"])
        node.GetDisplayNode().SetColor(*elec["color"])
        node.GetDisplayNode().SetGlyphScale(2.0)
        node.GetDisplayNode().SetTextScale(3.0)
        node.GetDisplayNode().SetVisibility(1)

        for i, (x, y, z) in enumerate(elec.get("coords", [])):
            idx = node.AddControlPoint(vtk.vtkVector3d(x, y, z))
            node.SetNthControlPointLabel(idx, f"{elec_id}{i+1}")

        self._nodes[elec_id] = node
        self._applyStatusColor(elec_id)

    def removeElectrode(self, elec_id: str):
        if elec_id in self._nodes:
            slicer.mrmlScene.RemoveNode(self._nodes.pop(elec_id))
        self._electrodes.pop(elec_id, None)

    def clearAll(self):
        for elec_id in list(self._nodes.keys()):
            self.removeElectrode(elec_id)
        self._electrodes.clear()
        self._colorIndex = 0

    # ─────────────────────────────────────────────────────────────────────────
    #  Getters
    # ─────────────────────────────────────────────────────────────────────────
    def getElectrode(self, elec_id: str) -> Optional[dict]:
        return self._electrodes.get(elec_id)

    def getAllElectrodes(self) -> List[dict]:
        return list(self._electrodes.values())

    # ─────────────────────────────────────────────────────────────────────────
    #  Visibilité & apparence
    # ─────────────────────────────────────────────────────────────────────────
    def setVisibility(self, elec_id: str, visible: bool):
        self._electrodes[elec_id]["visible"] = visible
        node = self._nodes.get(elec_id)
        if node:
            node.GetDisplayNode().SetVisibility(1 if visible else 0)

    def setGlobalOpacity(self, opacity: float):
        """opacity : 0.0 – 1.0"""
        for node in self._nodes.values():
            dn = node.GetDisplayNode()
            if dn:
                dn.SetOpacity(opacity)

    def setLabelsVisible(self, visible: bool):
        for node in self._nodes.values():
            dn = node.GetDisplayNode()
            if dn:
                dn.SetPointLabelsVisibility(visible)

    def setContactsNumberVisible(self, visible: bool):
        """Affiche/masque le numéro de contact sur chaque fiducial."""
        for elec_id, node in self._nodes.items():
            dn = node.GetDisplayNode()
            if dn:
                dn.SetPointLabelsVisibility(visible)

    def highlightElectrode(self, elec_id: str):
        """Met en surbrillance une électrode (taille augmentée, blanc)."""
        # Rétablit la précédente
        if self._highlighted and self._highlighted in self._nodes:
            self._applyStatusColor(self._highlighted)

        node = self._nodes.get(elec_id)
        if node:
            dn = node.GetDisplayNode()
            dn.SetSelectedColor(1.0, 1.0, 1.0)
            dn.SetGlyphScale(3.5)
            # Force la sélection de tous les points
            for i in range(node.GetNumberOfControlPoints()):
                node.SetNthControlPointSelected(i, True)

        self._highlighted = elec_id

    # ─────────────────────────────────────────────────────────────────────────
    #  Filtres
    # ─────────────────────────────────────────────────────────────────────────
    def applyConfidenceThreshold(self, threshold: float):
        """Masque les électrodes dont la confiance est < threshold."""
        self._confThreshold = threshold
        for elec_id, elec in self._electrodes.items():
            visible = elec.get("confidence", 0) >= threshold and elec.get("visible", True)
            node = self._nodes.get(elec_id)
            if node:
                node.GetDisplayNode().SetVisibility(1 if visible else 0)

    def applyStatusFilter(self, show: dict):
        """show = {"validated": bool, "average": bool, "low": bool}"""
        self._statusFilter = show
        for elec_id, elec in self._electrodes.items():
            status  = elec.get("status", "low")
            visible = show.get(status, True) and elec.get("visible", True)
            node = self._nodes.get(elec_id)
            if node:
                node.GetDisplayNode().SetVisibility(1 if visible else 0)

    # ─────────────────────────────────────────────────────────────────────────
    #  Interne
    # ─────────────────────────────────────────────────────────────────────────
    def _applyStatusColor(self, elec_id: str):
        """Restaure la couleur de statut d'une électrode."""
        elec = self._electrodes.get(elec_id)
        node = self._nodes.get(elec_id)
        if not elec or not node:
            return

        status_colors = {
            "validated": [0.30, 0.85, 0.30],
            "average":   [1.00, 0.60, 0.00],
            "low":       [0.95, 0.20, 0.20],
        }
        color = status_colors.get(elec.get("status", "low"), [1, 1, 1])
        dn = node.GetDisplayNode()
        if dn:
            dn.SetColor(*color)
            dn.SetSelectedColor(*color)
            dn.SetGlyphScale(2.0)
            for i in range(node.GetNumberOfControlPoints()):
                node.SetNthControlPointSelected(i, False)
