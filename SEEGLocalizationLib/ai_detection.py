import os
import numpy as np
import slicer

from typing import List
from scipy.ndimage import center_of_mass, label
import xml.etree.ElementTree as ET
from itertools import combinations

try:
    import onnxruntime as ort
except Exception:
    ort = None


class AIDetectionWorker:
    """
    Adaptateur entre le module SEEGLocalization et le vrai pipeline ONNX.

    Entrée réelle : un volume CT thresholded (ct_thresholded.nii.gz)
    Sortie pour l'UI actuelle : list[dict] au format attendu par ElectrodeManager
    """

    DEFAULT_MODEL_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "Resources", "model", "model_contacts.onnx")
    )
    DEFAULT_MODELS_XML_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "Resources", "config", "electrodeModels.xml")
    )

    def __init__(self, ctNode, mriNode=None, onnxPath=None, modelsXmlPath=None):
        self.ctNode = ctNode
        self.mriNode = mriNode  # conservé pour compatibilité UI ; non utilisé par le modèle actuel
        self.onnxPath = onnxPath or self.DEFAULT_MODEL_PATH
        self.modelsXmlPath = modelsXmlPath or self.DEFAULT_MODELS_XML_PATH

        self.patch_size = (48, 48, 48)
        self.overlap = 0.5
        self.prob_threshold = 0.5
        self.min_component_voxels = 3
        self.default_confidence = 0.85

    def run(self, progressCallback=None) -> List[dict]:
        def _progress(percent, message=""):
            if progressCallback:
                progressCallback(percent, message)

        _progress(5, "Vérification des entrées…")
        self._validateInputs()

        _progress(10, "Chargement du volume CT thresholded…")
        contacts, orderedElectrodes = self._runRealInference(_progress)

        _progress(92, "Adaptation des résultats à l'interface…")
        electrodes = self._convertToTemplateElectrodes(orderedElectrodes)

        _progress(100, f"Détection terminée — {len(electrodes)} électrode(s)")
        return electrodes

    def _validateInputs(self):
        if self.ctNode is None:
            raise ValueError("Volume CT thresholded non sélectionné.")

        if ort is None:
            raise RuntimeError(
                "onnxruntime n'est pas installé dans le Python de Slicer."
            )

        if not self.onnxPath or not os.path.isfile(self.onnxPath):
            raise FileNotFoundError(f"Fichier ONNX introuvable : {self.onnxPath}")

        # XML optionnel : si absent, on utilisera les espacements par défaut
        if self.modelsXmlPath and (not os.path.exists(self.modelsXmlPath)):
            self.modelsXmlPath = None

    def _runRealInference(self, progressCallback):
        volumeArray = slicer.util.arrayFromVolume(self.ctNode)  # shape: z,y,x
        volumeArray = np.transpose(volumeArray, (2, 1, 0)).astype(np.float32)  # -> x,y,z

        # Le modèle a été entraîné sur un volume binaire thresholded
        volumeArray = (volumeArray > 0).astype(np.float32)

        progressCallback(20, "Inférence ONNX en cours…")
        prob_map = self.runONNXInference(volumeArray, self.onnxPath, progressCallback)
        predMask = (prob_map >= self.prob_threshold).astype(np.uint8)

        progressCallback(75, "Extraction des centres de contacts…")
        affine = self.getIJKToRASMatrixAsNumpy(self.ctNode)
        contacts = self.extractContactCenters(predMask, affine)

        progressCallback(82, "Regroupement des contacts en électrodes…")
        plausible_spacings = self.loadPlausibleSpacings(self.modelsXmlPath)
        electrodes, leftovers = self.groupContactsIntoElectrodes(
            contacts,
            plausible_spacings=plausible_spacings,
        )

        progressCallback(88, "Ordonnancement des contacts…")
        orderedElectrodes = self.orderContacts(electrodes)

        return contacts, orderedElectrodes

    def runONNXInference(self, volumeArray, onnxPath, progressCallback=None):
        session = ort.InferenceSession(onnxPath, providers=["CPUExecutionProvider"])

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        sx, sy, sz = volumeArray.shape
        px, py, pz = self.patch_size

        stride_x = max(1, int(px * (1 - self.overlap)))
        stride_y = max(1, int(py * (1 - self.overlap)))
        stride_z = max(1, int(pz * (1 - self.overlap)))

        xs = self.getStarts(sx, px, stride_x)
        ys = self.getStarts(sy, py, stride_y)
        zs = self.getStarts(sz, pz, stride_z)

        total_patches = len(xs) * len(ys) * len(zs)
        done = 0

        prob_acc = np.zeros((sx, sy, sz), dtype=np.float32)
        count_acc = np.zeros((sx, sy, sz), dtype=np.float32)

        for x0 in xs:
            for y0 in ys:
                for z0 in zs:
                    patch = volumeArray[x0:x0+px, y0:y0+py, z0:z0+pz]
                    patch = patch[None, None, ...].astype(np.float32)  # (1,1,x,y,z)

                    outputs = session.run([output_name], {input_name: patch})
                    logits = outputs[0][0, 0]
                    probs = 1.0 / (1.0 + np.exp(-logits))

                    prob_acc[x0:x0+px, y0:y0+py, z0:z0+pz] += probs
                    count_acc[x0:x0+px, y0:y0+py, z0:z0+pz] += 1.0

                    done += 1
                    if progressCallback and total_patches > 0:
                        pct = 20 + int(50 * done / total_patches)
                        progressCallback(pct, f"Inférence ONNX… patch {done}/{total_patches}")

        prob_map = prob_acc / np.maximum(count_acc, 1e-6)
        return prob_map

    def getStarts(self, dim, patch, stride):
        if dim <= patch:
            return [0]
        starts = list(range(0, dim - patch + 1, stride))
        if starts[-1] != dim - patch:
            starts.append(dim - patch)
        return starts

    def extractContactCenters(self, predMask, affine):
        labeled, ncomp = label(predMask.astype(np.uint8))

        contacts = []
        cid = 1

        for k in range(1, ncomp + 1):
            comp = (labeled == k)
            size = int(comp.sum())
            if size < self.min_component_voxels:
                continue

            com = center_of_mass(comp.astype(np.uint8))
            voxel_center = [float(com[0]), float(com[1]), float(com[2])]
            world_center = self.voxelToWorld(affine, voxel_center).tolist()

            contacts.append({
                "id": cid,
                "component_label": int(k),
                "n_voxels": size,
                "voxel_center": voxel_center,
                "world_center_mm": world_center,
            })
            cid += 1

        return contacts

    def voxelToWorld(self, affine, ijk):
        ijk_h = np.array([ijk[0], ijk[1], ijk[2], 1.0], dtype=np.float64)
        xyz = affine @ ijk_h
        return xyz[:3]

    def loadPlausibleSpacings(self, xmlPath):
        default_spacings = [1.0, 2.0, 3.5, 3.9, 4.0, 4.8, 6.1, 7.6]

        if xmlPath is None or not os.path.exists(xmlPath):
            return default_spacings

        try:
            tree = ET.parse(xmlPath)
            root = tree.getroot()
            spacings = set()

            for elem in root.findall(".//ElectrodeModel"):
                if "contactSpacing" in elem.attrib:
                    try:
                        spacings.add(float(elem.attrib["contactSpacing"]))
                    except Exception:
                        pass
                if "firstContactSpacing" in elem.attrib:
                    try:
                        spacings.add(float(elem.attrib["firstContactSpacing"]))
                    except Exception:
                        pass

            spacings = sorted([s for s in spacings if s > 0])
            return spacings if spacings else default_spacings
        except Exception:
            return default_spacings

    def fitLine(self, points):
        center = points.mean(axis=0)
        _, _, vh = np.linalg.svd(points - center[None, :], full_matrices=False)
        direction = vh[0]
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        return center, direction

    def pointLineDistance(self, points, line_point, line_dir):
        vecs = points - line_point[None, :]
        proj = np.sum(vecs * line_dir[None, :], axis=1, keepdims=True) * line_dir[None, :]
        perp = vecs - proj
        return np.linalg.norm(perp, axis=1)

    def projectionOnLine(self, points, line_point, line_dir):
        return np.dot(points - line_point[None, :], line_dir)

    def plausibleGapPenalty(self, gaps, plausible_spacings, max_multiple=3):
        if len(gaps) == 0:
            return 0.0
        candidates = []
        for s in plausible_spacings:
            for k in range(1, max_multiple + 1):
                candidates.append(k * s)
        penalties = []
        for g in gaps:
            penalties.append(min(abs(g - c) for c in candidates))
        return float(np.mean(penalties))

    def buildBestOrderedChain(self, sorted_proj, sorted_points, sorted_ids, plausible_spacings,
                              min_contacts=3, min_gap=1.0, max_gap=20.0):
        n = len(sorted_proj)
        if n < min_contacts:
            return [], [], []

        best_score = -1e9
        best_slice = None

        for i in range(n):
            for j in range(i + min_contacts - 1, n):
                sub_proj = sorted_proj[i:j+1]
                gaps = np.diff(sub_proj)

                if len(gaps) > 0:
                    if np.any(gaps < min_gap) or np.any(gaps > max_gap):
                        continue

                spacing_pen = self.plausibleGapPenalty(gaps, plausible_spacings)
                score = len(sub_proj) - 0.25 * spacing_pen

                if score > best_score:
                    best_score = score
                    best_slice = (i, j)

        if best_slice is None:
            return [], [], []

        i, j = best_slice
        return sorted_proj[i:j+1], sorted_points[i:j+1], sorted_ids[i:j+1]

    def groupContactsIntoElectrodes(self, contacts, plausible_spacings):
        if len(contacts) == 0:
            return [], []

        points = np.array([c["world_center_mm"] for c in contacts], dtype=np.float64)
        point_ids = [c["id"] for c in contacts]

        remaining_points = points.copy()
        remaining_ids = point_ids.copy()

        electrodes = []
        electrode_id = 1

        line_tol_mm = 2.0
        min_contacts_per_electrode = 3
        min_seed_dist_mm = 2.0
        max_seed_dist_mm = 18.0
        min_gap_mm = 1.0
        max_gap_mm = 20.0

        while len(remaining_points) >= min_contacts_per_electrode:
            best_group_indices = None
            best_center = None
            best_dir = None
            best_score = -1e9

            pair_indices = list(combinations(range(len(remaining_points)), 2))

            for i, j in pair_indices:
                p1 = remaining_points[i]
                p2 = remaining_points[j]

                seed_dist = np.linalg.norm(p2 - p1)
                if seed_dist < min_seed_dist_mm or seed_dist > max_seed_dist_mm:
                    continue

                direction = p2 - p1
                direction = direction / (np.linalg.norm(direction) + 1e-8)

                dists = self.pointLineDistance(remaining_points, p1, direction)
                inlier_idx = np.where(dists <= line_tol_mm)[0]

                if len(inlier_idx) < min_contacts_per_electrode:
                    continue

                inlier_points = remaining_points[inlier_idx]
                center, refined_dir = self.fitLine(inlier_points)

                refined_dists = self.pointLineDistance(remaining_points, center, refined_dir)
                refined_idx = np.where(refined_dists <= line_tol_mm)[0]

                if len(refined_idx) < min_contacts_per_electrode:
                    continue

                refined_points = remaining_points[refined_idx]
                refined_ids = [remaining_ids[k] for k in refined_idx]

                proj = self.projectionOnLine(refined_points, center, refined_dir)
                order = np.argsort(proj)

                sorted_proj = proj[order]
                sorted_points = refined_points[order]
                sorted_ids = [refined_ids[k] for k in order]

                chain_proj, chain_points, chain_ids = self.buildBestOrderedChain(
                    sorted_proj=sorted_proj,
                    sorted_points=sorted_points,
                    sorted_ids=sorted_ids,
                    plausible_spacings=plausible_spacings,
                    min_contacts=min_contacts_per_electrode,
                    min_gap=min_gap_mm,
                    max_gap=max_gap_mm,
                )

                if len(chain_ids) < min_contacts_per_electrode:
                    continue

                chain_gaps = np.diff(chain_proj) if len(chain_proj) > 1 else np.array([])
                spacing_pen = self.plausibleGapPenalty(chain_gaps, plausible_spacings)
                perp_d = self.pointLineDistance(np.array(chain_points), center, refined_dir).mean()

                score = len(chain_ids) - 0.20 * spacing_pen - 0.20 * perp_d

                if score > best_score:
                    best_score = score
                    best_center = center
                    best_dir = refined_dir
                    best_group_indices = [remaining_ids.index(cid) for cid in chain_ids]

            if best_group_indices is None:
                break

            group_points = remaining_points[best_group_indices]
            group_ids = [remaining_ids[idx] for idx in best_group_indices]

            proj = self.projectionOnLine(group_points, best_center, best_dir)
            order = np.argsort(proj)

            ordered_points = group_points[order]
            ordered_ids = [group_ids[k] for k in order]

            electrodes.append({
                "electrode_id": electrode_id,
                "n_contacts": len(ordered_ids),
                "axis_point_mm": best_center.tolist(),
                "axis_direction": best_dir.tolist(),
                "contact_ids": ordered_ids,
                "ordered_world_centers_mm": ordered_points.tolist(),
            })

            keep_mask = np.ones(len(remaining_points), dtype=bool)
            keep_mask[best_group_indices] = False
            remaining_points = remaining_points[keep_mask]
            remaining_ids = [rid for k, rid in enumerate(remaining_ids) if keep_mask[k]]

            electrode_id += 1

        leftovers = []
        for rid, p in zip(remaining_ids, remaining_points):
            leftovers.append({
                "contact_id": rid,
                "world_center_mm": p.tolist(),
            })

        return electrodes, leftovers

    def orderContacts(self, electrodes):
        orderedElectrodes = []

        for e_idx, elec in enumerate(electrodes, start=1):
            electrode_name_auto = f"E{e_idx}"
            ordered_points = elec.get("ordered_world_centers_mm", [])

            ordered_contacts = []
            for c_idx, xyz in enumerate(ordered_points, start=1):
                ordered_contacts.append({
                    "contact_order": c_idx,
                    "contact_name_auto": f"{electrode_name_auto}_{c_idx}",
                    "world_center_mm": xyz,
                })

            orderedElectrodes.append({
                "electrode_id": elec["electrode_id"],
                "electrode_name_auto": electrode_name_auto,
                "n_contacts": len(ordered_contacts),
                "axis_point_mm": elec.get("axis_point_mm"),
                "axis_direction": elec.get("axis_direction"),
                "ordered_contacts": ordered_contacts,
            })

        return orderedElectrodes

    def _convertToTemplateElectrodes(self, orderedElectrodes):
        electrodes = []
        for elec in orderedElectrodes:
            coords = []
            for c in elec.get("ordered_contacts", []):
                xyz = c.get("world_center_mm", [])
                if len(xyz) == 3:
                    coords.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])

            electrodes.append({
                "id": elec.get("electrode_name_auto", f"E{len(electrodes)+1}"),
                "contacts": int(elec.get("n_contacts", len(coords))),
                "confidence": float(self.default_confidence),
                "coords": coords,
            })
        return electrodes

    def getIJKToRASMatrixAsNumpy(self, volumeNode):
        import vtk
        m = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(m)
        arr = np.eye(4, dtype=np.float64)
        for i in range(4):
            for j in range(4):
                arr[i, j] = m.GetElement(i, j)
        return arr