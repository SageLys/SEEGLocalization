"""
ExportManager
Gère l'export des électrodes SEEG détectées dans différents formats.

Formats supportés :
  - csv  : tableau plat (électrode, contact, X, Y, Z, confiance, statut)
  - json : structure complète avec toutes les métadonnées
  - bids : format BIDS ieeg (_electrodes.tsv + _coordsystem.json)
  - mrml : sauvegarde de la scène 3D Slicer complète (.mrb)
"""

import os
import csv
import json
import datetime
import slicer


class ExportManager:
    """Exporte les données d'électrodes dans le format demandé."""

    def __init__(self, electrodeManager):
        self.manager = electrodeManager

    # ─────────────────────────────────────────────────────────────────────────
    def export(self, folder: str, options: dict) -> str:
        """
        Point d'entrée principal.
        Retourne le chemin du fichier créé.
        """
        fmt = options.get("format", "csv")
        electrodes = self._filterElectrodes(options)

        dispatch = {
            "csv":  self._exportCSV,
            "json": self._exportJSON,
            "bids": self._exportBIDS,
            "mrml": self._exportMRML,
        }

        handler = dispatch.get(fmt)
        if not handler:
            raise ValueError(f"Format inconnu : {fmt}")

        return handler(folder, electrodes, options)

    # ─────────────────────────────────────────────────────────────────────────
    #  CSV
    # ─────────────────────────────────────────────────────────────────────────
    def _exportCSV(self, folder: str, electrodes: list, options: dict) -> str:
        pid  = options.get("patient_id", "Patient")
        path = os.path.join(folder, f"{pid}_SEEG_electrodes.csv")

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # En-tête
            header = ["electrode_id", "contact_index", "x_mm", "y_mm", "z_mm", "status"]
            if options.get("include_conf"):
                header.append("confidence")
            writer.writerow(header)

            for elec in electrodes:
                for i, (x, y, z) in enumerate(elec.get("coords", [])):
                    row = [
                        elec["id"],
                        i + 1,
                        f"{x:.4f}", f"{y:.4f}", f"{z:.4f}",
                        elec.get("status", "unknown"),
                    ]
                    if options.get("include_conf"):
                        row.append(f"{elec.get('confidence', 0):.4f}")
                    writer.writerow(row)

        return path

    # ─────────────────────────────────────────────────────────────────────────
    #  JSON
    # ─────────────────────────────────────────────────────────────────────────
    def _exportJSON(self, folder: str, electrodes: list, options: dict) -> str:
        pid  = options.get("patient_id", "Patient")
        path = os.path.join(folder, f"{pid}_SEEG_electrodes.json")

        payload = {
            "patient_id":    pid,
            "export_date":   datetime.datetime.now().isoformat(),
            "software":      "3D Slicer — SEEG Localization Module G05",
            "coord_system":  "RAS_mm",
            "electrodes":    []
        }

        for elec in electrodes:
            entry = {
                "id":       elec["id"],
                "contacts": elec.get("contacts", len(elec.get("coords", []))),
                "status":   elec.get("status", "unknown"),
                "coords":   [[round(x, 4), round(y, 4), round(z, 4)]
                             for x, y, z in elec.get("coords", [])],
            }
            if options.get("include_conf"):
                entry["confidence"] = round(elec.get("confidence", 0), 4)
            payload["electrodes"].append(entry)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return path

    # ─────────────────────────────────────────────────────────────────────────
    #  BIDS ieeg
    # ─────────────────────────────────────────────────────────────────────────
    def _exportBIDS(self, folder: str, electrodes: list, options: dict) -> str:
        """
        Génère :
          sub-<pid>_electrodes.tsv
          sub-<pid>_coordsystem.json
        conforme BIDS 1.7 ieeg.
        """
        pid = options.get("patient_id", "Patient").replace("_", "")

        # ── TSV ──────────────────────────────────────────────────────────────
        tsv_path = os.path.join(folder, f"sub-{pid}_electrodes.tsv")
        with open(tsv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            header = ["name", "x", "y", "z", "size", "type", "status"]
            if options.get("include_conf"):
                header.append("confidence")
            writer.writerow(header)

            for elec in electrodes:
                for i, (x, y, z) in enumerate(elec.get("coords", [])):
                    name = f"{elec['id']}{i+1:02d}"
                    row  = [name,
                            f"{x:.4f}", f"{y:.4f}", f"{z:.4f}",
                            "2.0",      # diamètre en mm
                            "SEEG",
                            elec.get("status", "n/a")]
                    if options.get("include_conf"):
                        row.append(f"{elec.get('confidence', 0):.4f}")
                    writer.writerow(row)

        # ── coordsystem.json ─────────────────────────────────────────────────
        coord_path = os.path.join(folder, f"sub-{pid}_coordsystem.json")
        coord_sys  = {
            "iEEGCoordinateSystem":          "RAS",
            "iEEGCoordinateUnits":           "mm",
            "iEEGCoordinateSystemDescription":
                "Coordinates in RAS space, referenced to MRI T1 pre-operative scan.",
            "IntendedFor": f"anat/sub-{pid}_T1w.nii.gz"
        }
        with open(coord_path, "w", encoding="utf-8") as f:
            json.dump(coord_sys, f, indent=2)

        return tsv_path     # fichier principal

    # ─────────────────────────────────────────────────────────────────────────
    #  MRML / MRB (scène Slicer)
    # ─────────────────────────────────────────────────────────────────────────
    def _exportMRML(self, folder: str, electrodes: list, options: dict) -> str:
        pid  = options.get("patient_id", "Patient")
        path = os.path.join(folder, f"{pid}_SEEG_scene.mrb")
        slicer.util.saveScene(path)
        return path

    # ─────────────────────────────────────────────────────────────────────────
    #  Filtrage
    # ─────────────────────────────────────────────────────────────────────────
    def _filterElectrodes(self, options: dict) -> list:
        electrodes = self.manager.getAllElectrodes()
        if options.get("only_validated"):
            electrodes = [e for e in electrodes if e.get("status") == "validated"]
        return electrodes
