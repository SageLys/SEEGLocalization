"""
SEEG Localization Module (G05)
Module 3D Slicer pour la localisation automatique des électrodes SEEG
via un modèle IA 3D U-Net sur des images MRI T1 + CT post-opératoire.
"""

import os
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import vtk
import numpy as np

from SEEGLocalizationLib.electrode_manager import ElectrodeManager
from SEEGLocalizationLib.ai_detection import AIDetectionWorker
from SEEGLocalizationLib.export_manager import ExportManager


# ─────────────────────────────────────────────────────────────────────────────
#  Metadata du module
# ─────────────────────────────────────────────────────────────────────────────
class SEEGLocalization(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title       = "SEEG Localization Module (G05)"
        self.parent.categories  = ["Epilepsy", "SEEG"]
        self.parent.dependencies = []
        self.parent.contributors = ["G05 Team"]
        self.parent.helpText    = "Localisation automatique des électrodes SEEG par IA (3D U-Net)."
        self.parent.acknowledgementText = "Module développé dans le cadre du projet G05."


# ─────────────────────────────────────────────────────────────────────────────
#  Widget principal (IHM)
# ─────────────────────────────────────────────────────────────────────────────
class SEEGLocalizationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic            = None
        self.electrodeManager = None
        self._electrodeRows   = {}   # id → row index dans la table

    # ── Setup général ────────────────────────────────────────────────────────
    def setup(self):
        super().setup()
        self.logic            = SEEGLocalizationLogic()
        self.electrodeManager = ElectrodeManager()

        # ── Layout 4 vues (Axial / Sagittal / Coronal / 3D) ─────────────────
        self._setup4UpLayout()

        # ── Panel gauche : onglets ────────────────────────────────────────────
        self.tabWidget = qt.QTabWidget()
        self.tabWidget.setStyleSheet("""
            QTabWidget::pane  { border: none; background: #1e1e1e; }
            QTabBar::tab      { background: #2a2a2a; color: #aaa;
                                padding: 8px 18px; border-radius: 4px 4px 0 0; }
            QTabBar::tab:selected { background: #3a7bd5; color: white; font-weight: bold; }
            
            /* --- UI Readability Fix (Black text on black background) --- */
            QLabel, QCheckBox, QRadioButton {
                color: white;
            }
            QLineEdit, QComboBox, qMRMLNodeComboBox {
                background-color: #2a2a2a;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 4px;
            }
            QComboBox::drop-down {
                border-left: 1px solid #555;
            }
            QComboBox QAbstractItemView {
                background-color: #2a2a2a;
                color: white;
                selection-background-color: #3a7bd5;
            }
        """)

        self._buildDataTab()
        self._buildElectrodeTab()
        self._buildExportTab()

        self.layout.addWidget(self.tabWidget)
        self.layout.addStretch()

        # ── Observers MRML ───────────────────────────────────────────────────
        self.addObserver(slicer.mrmlScene,
                         slicer.mrmlScene.NodeAddedEvent,
                         self._onNodeAdded)

    # ── Layout 4 vues ────────────────────────────────────────────────────────
    def _setup4UpLayout(self):
        layoutManager = slicer.app.layoutManager()
        # Layout ID 3 = Four-Up (Axial/Sagittal/Coronal/3D)
        layoutManager.setLayout(3)
        # Couleur de fond 3D → noir
        view3D = layoutManager.threeDWidget(0).threeDView()
        view3D.setBackgroundColor(qt.QColor(30, 30, 30))
        view3D.setBackgroundColor2(qt.QColor(30, 30, 30))

    # ─────────────────────────────────────────────────────────────────────────
    #  ONGLET DATA
    # ─────────────────────────────────────────────────────────────────────────
    def _buildDataTab(self):
        tab = qt.QWidget()
        layout = qt.QVBoxLayout(tab)
        layout.setContentsMargins(12, 14, 12, 14)
        layout.setSpacing(10)

        # ── Section Selection ────────────────────────────────────────────────
        selBox = ctk.ctkCollapsibleGroupBox()
        selBox.title = "Selection"
        selBox.setStyleSheet("QGroupBox { color: white; font-weight: bold; }")
        selLayout = qt.QFormLayout(selBox)

        # MRI T1
        self.mriSelector = slicer.qMRMLNodeComboBox()
        self.mriSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.mriSelector.selectNodeUponCreation = True
        self.mriSelector.addEnabled  = False
        self.mriSelector.removeEnabled = False
        self.mriSelector.noneEnabled = False
        self.mriSelector.showHidden  = False
        self.mriSelector.setMRMLScene(slicer.mrmlScene)
        self.mriSelector.setToolTip("Sélectionner le volume MRI T1 pré-opératoire")
        selLayout.addRow("MRI T1 Volume", self.mriSelector)

        # CT Post-op
        self.ctSelector = slicer.qMRMLNodeComboBox()
        self.ctSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.ctSelector.selectNodeUponCreation = False
        self.ctSelector.addEnabled  = False
        self.ctSelector.removeEnabled = False
        self.ctSelector.noneEnabled = True
        self.ctSelector.showHidden  = False
        self.ctSelector.setMRMLScene(slicer.mrmlScene)
        self.ctSelector.setToolTip("Sélectionner le volume CT post-opératoire")
        selLayout.addRow("Post-Op CT Volume", self.ctSelector)

        layout.addWidget(selBox)

        # ── Section AI Detection ─────────────────────────────────────────────
        aiBox = ctk.ctkCollapsibleGroupBox()
        aiBox.title = "AI Detection"
        aiBox.setStyleSheet("QGroupBox { color: white; font-weight: bold; }")
        aiLayout = qt.QVBoxLayout(aiBox)

        self.runAIButton = qt.QPushButton("Run AI Detection (3D U-Net)")
        self.runAIButton.setStyleSheet("""
            QPushButton {
                background: #3a7bd5; color: white; font-weight: bold;
                border-radius: 6px; padding: 10px;
            }
            QPushButton:hover   { background: #2e6bc5; }
            QPushButton:pressed { background: #1e5ab5; }
            QPushButton:disabled{ background: #555; color: #888; }
        """)
        self.runAIButton.clicked.connect(self._onRunAIDetection)
        aiLayout.addWidget(self.runAIButton)

        self.progressBar = qt.QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        self.progressBar.setVisible(False)
        self.progressBar.setStyleSheet("""
            QProgressBar { border-radius: 4px; background: #333; color: white; text-align: center; }
            QProgressBar::chunk { background: #4caf50; border-radius: 4px; }
        """)
        aiLayout.addWidget(self.progressBar)

        self.statusLabel = qt.QLabel("")
        self.statusLabel.setStyleSheet("color: #4caf50; font-size: 12px;")
        self.statusLabel.setAlignment(qt.Qt.AlignCenter)
        aiLayout.addWidget(self.statusLabel)

        layout.addWidget(aiBox)

        # ── Charger volumes depuis le disque ─────────────────────────────────
        loadBox = ctk.ctkCollapsibleGroupBox()
        loadBox.title = "Charger depuis le disque"
        loadBox.collapsed = False
        loadBox.setStyleSheet("QGroupBox { color: white; font-weight: bold; }")
        loadLayout = qt.QVBoxLayout(loadBox)

        self.loadMRIButton = qt.QPushButton("📂  Charger MRI T1 (.nii / .nii.gz)")
        self.loadMRIButton.setStyleSheet(self._btnStyle("#555"))
        self.loadMRIButton.clicked.connect(self._onLoadMRI)
        loadLayout.addWidget(self.loadMRIButton)

        self.loadCTButton = qt.QPushButton("📂  Charger CT Post-Op (.nii / .nii.gz)")
        self.loadCTButton.setStyleSheet(self._btnStyle("#555"))
        self.loadCTButton.clicked.connect(self._onLoadCT)
        loadLayout.addWidget(self.loadCTButton)

        layout.addWidget(loadBox)
        layout.addStretch()

        self.tabWidget.addTab(tab, "Data")

    # ─────────────────────────────────────────────────────────────────────────
    #  ONGLET ELECTRODE
    # ─────────────────────────────────────────────────────────────────────────
    def _buildElectrodeTab(self):
        tab = qt.QWidget()
        layout = qt.QVBoxLayout(tab)
        layout.setContentsMargins(12, 14, 12, 14)
        layout.setSpacing(10)

        # ── Transparence ─────────────────────────────────────────────────────
        transBox = ctk.ctkCollapsibleGroupBox()
        transBox.title = "Transparency"
        transBox.setStyleSheet("QGroupBox { color: white; font-weight: bold; }")
        transLayout = qt.QFormLayout(transBox)

        self.brainOpacitySlider = ctk.ctkSliderWidget()
        self.brainOpacitySlider.minimum = 0
        self.brainOpacitySlider.maximum = 100
        self.brainOpacitySlider.value   = 86
        self.brainOpacitySlider.suffix  = "%"
        self.brainOpacitySlider.singleStep = 1
        self.brainOpacitySlider.valueChanged.connect(self._onBrainOpacityChanged)
        transLayout.addRow("Brain Opacity", self.brainOpacitySlider)

        self.electrodeOpacitySlider = ctk.ctkSliderWidget()
        self.electrodeOpacitySlider.minimum = 0
        self.electrodeOpacitySlider.maximum = 100
        self.electrodeOpacitySlider.value   = 100
        self.electrodeOpacitySlider.suffix  = "%"
        self.electrodeOpacitySlider.singleStep = 1
        self.electrodeOpacitySlider.valueChanged.connect(self._onElectrodeOpacityChanged)
        transLayout.addRow("Electrodes Opacity", self.electrodeOpacitySlider)

        layout.addWidget(transBox)

        # ── Labels ───────────────────────────────────────────────────────────
        labelsBox = ctk.ctkCollapsibleGroupBox()
        labelsBox.title = "Labels"
        labelsBox.setStyleSheet("QGroupBox { color: white; font-weight: bold; }")
        labelsLayout = qt.QFormLayout(labelsBox)

        self.showLabelsCheck = qt.QCheckBox()
        self.showLabelsCheck.setChecked(False)
        self.showLabelsCheck.stateChanged.connect(self._onShowLabelsChanged)
        labelsLayout.addRow("Show electrodes labels", self.showLabelsCheck)

        self.showContactsCheck = qt.QCheckBox()
        self.showContactsCheck.setChecked(False)
        self.showContactsCheck.stateChanged.connect(self._onShowContactsChanged)
        labelsLayout.addRow("Show number contacts", self.showContactsCheck)

        layout.addWidget(labelsBox)

        # ── Liste des électrodes ─────────────────────────────────────────────
        listBox = ctk.ctkCollapsibleGroupBox()
        listBox.title = "Electrodes List"
        listBox.setStyleSheet("QGroupBox { color: white; font-weight: bold; }")
        listLayout = qt.QVBoxLayout(listBox)

        self.electrodeTable = qt.QTableWidget()
        self.electrodeTable.setColumnCount(5)
        self.electrodeTable.setHorizontalHeaderLabels(
            ["Vis.", "ID", "Contacts", "Conf.", "Status"])
        self.electrodeTable.horizontalHeader().setStretchLastSection(True)
        self.electrodeTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.electrodeTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.electrodeTable.setStyleSheet("""
            QTableWidget { background: #1a1a1a; color: white;
                           gridline-color: #333; border: none; }
            QTableWidget::item:selected { background: #3a7bd5; }
            QHeaderView::section { background: #2a2a2a; color: #aaa;
                                   padding: 4px; border: 1px solid #333; }
        """)
        self.electrodeTable.setColumnWidth(0, 35)
        self.electrodeTable.setColumnWidth(1, 35)
        self.electrodeTable.setColumnWidth(2, 65)
        self.electrodeTable.setColumnWidth(3, 50)
        self.electrodeTable.verticalHeader().setVisible(False)
        self.electrodeTable.itemSelectionChanged.connect(self._onElectrodeSelected)
        listLayout.addWidget(self.electrodeTable)

        # ── Info coordonnées de l'électrode sélectionnée ─────────────────────
        self.coordsLabel = qt.QLabel("Sélectionner une électrode")
        self.coordsLabel.setStyleSheet("""
            color: #aaa; font-size: 11px; padding: 6px;
            background: #1a1a1a; border-radius: 4px;
        """)
        self.coordsLabel.setAlignment(qt.Qt.AlignCenter)
        listLayout.addWidget(self.coordsLabel)

        layout.addWidget(listBox)

        # ── Seuil de confiance ───────────────────────────────────────────────
        threshBox = ctk.ctkCollapsibleGroupBox()
        threshBox.title = "Filtrage"
        threshBox.setStyleSheet("QGroupBox { color: white; font-weight: bold; }")
        threshLayout = qt.QFormLayout(threshBox)

        self.confThreshSlider = ctk.ctkSliderWidget()
        self.confThreshSlider.minimum = 0
        self.confThreshSlider.maximum = 100
        self.confThreshSlider.value   = 92
        self.confThreshSlider.suffix  = "%"
        self.confThreshSlider.singleStep = 1
        self.confThreshSlider.valueChanged.connect(self._onConfidenceThresholdChanged)
        threshLayout.addRow("Minimum confidence threshold", self.confThreshSlider)

        layout.addWidget(threshBox)

        # ── Filtres statut ───────────────────────────────────────────────────
        statusBox = ctk.ctkCollapsibleGroupBox()
        statusBox.title = "Show by status"
        statusBox.setStyleSheet("QGroupBox { color: white; font-weight: bold; }")
        statusLayout = qt.QHBoxLayout(statusBox)

        self.showValidatedCheck    = self._coloredCheckbox("Validated",    "#4caf50")
        self.showAvgConfCheck      = self._coloredCheckbox("Average Conf.", "#ff9800")
        self.showLowConfCheck      = self._coloredCheckbox("Low Conf.",    "#f44336")
        statusLayout.addWidget(self.showValidatedCheck)
        statusLayout.addWidget(self.showAvgConfCheck)
        statusLayout.addWidget(self.showLowConfCheck)
        for cb in [self.showValidatedCheck, self.showAvgConfCheck, self.showLowConfCheck]:
            cb.setChecked(True)
            cb.stateChanged.connect(self._onStatusFilterChanged)

        layout.addWidget(statusBox)
        layout.addStretch()

        self.tabWidget.addTab(tab, "Electrode")

    # ─────────────────────────────────────────────────────────────────────────
    #  ONGLET EXPORT
    # ─────────────────────────────────────────────────────────────────────────
    def _buildExportTab(self):
        tab = qt.QWidget()
        layout = qt.QVBoxLayout(tab)
        layout.setContentsMargins(12, 14, 12, 14)
        layout.setSpacing(12)

        # Format
        fmtBox = ctk.ctkCollapsibleGroupBox()
        fmtBox.title = "Format d'export"
        fmtBox.setStyleSheet("QGroupBox { color: white; font-weight: bold; }")
        fmtLayout = qt.QVBoxLayout(fmtBox)

        self.exportCSV  = qt.QRadioButton("CSV  (coordonnées + métadonnées)")
        self.exportJSON = qt.QRadioButton("JSON (structure complète)")
        self.exportBIDS = qt.QRadioButton("BIDS ieeg (format recherche)")
        self.exportMRML = qt.QRadioButton("Scène MRML (3D Slicer natif)")
        self.exportCSV.setChecked(True)
        for rb in [self.exportCSV, self.exportJSON, self.exportBIDS, self.exportMRML]:
            rb.setStyleSheet("color: white;")
            fmtLayout.addWidget(rb)

        layout.addWidget(fmtBox)

        # Options
        optBox = ctk.ctkCollapsibleGroupBox()
        optBox.title = "Options"
        optBox.setStyleSheet("QGroupBox { color: white; font-weight: bold; }")
        optLayout = qt.QFormLayout(optBox)

        self.includeConfCheck = qt.QCheckBox()
        self.includeConfCheck.setChecked(True)
        optLayout.addRow("Inclure score de confiance", self.includeConfCheck)

        self.onlyValidatedCheck = qt.QCheckBox()
        self.onlyValidatedCheck.setChecked(False)
        optLayout.addRow("Exporter uniquement les validées", self.onlyValidatedCheck)

        self.patientIDEdit = qt.QLineEdit("Patient_001")
        optLayout.addRow("Patient ID", self.patientIDEdit)

        layout.addWidget(optBox)

        # Bouton export
        self.exportButton = qt.QPushButton("💾  Exporter les électrodes")
        self.exportButton.setStyleSheet("""
            QPushButton {
                background: #4caf50; color: white; font-weight: bold;
                border-radius: 6px; padding: 10px; font-size: 13px;
            }
            QPushButton:hover   { background: #43a047; }
            QPushButton:pressed { background: #388e3c; }
        """)
        self.exportButton.clicked.connect(self._onExport)
        layout.addWidget(self.exportButton)

        self.exportStatusLabel = qt.QLabel("")
        self.exportStatusLabel.setStyleSheet("color: #4caf50; font-size: 12px;")
        self.exportStatusLabel.setAlignment(qt.Qt.AlignCenter)
        layout.addWidget(self.exportStatusLabel)

        layout.addStretch()
        self.tabWidget.addTab(tab, "Export")

    # ─────────────────────────────────────────────────────────────────────────
    #  CALLBACKS – DATA
    # ─────────────────────────────────────────────────────────────────────────
    def _onLoadMRI(self):
        path = qt.QFileDialog.getOpenFileName(
            None, "Charger MRI T1", "", "NIfTI (*.nii *.nii.gz)")
        if path:
            node = slicer.util.loadVolume(path)
            slicer.util.setSliceViewerLayers(background=node)
            self.mriSelector.setCurrentNode(node)
            self.statusLabel.setStyleSheet("color: #4caf50; font-size: 12px;")
            self.statusLabel.setText(f"✔  MRI chargé : {os.path.basename(path)}")

    def _onLoadCT(self):
        path = qt.QFileDialog.getOpenFileName(
            None, "Charger CT Post-Op", "", "NIfTI (*.nii *.nii.gz)")
        if path:
            node = slicer.util.loadVolume(path)
            slicer.util.setSliceViewerLayers(foreground=node, foregroundOpacity=0.5)
            self.ctSelector.setCurrentNode(node)
            self.statusLabel.setStyleSheet("color: #4caf50; font-size: 12px;")
            self.statusLabel.setText(f"✔  CT chargé : {os.path.basename(path)}")

    def _onRunAIDetection(self):
        mriNode = self.mriSelector.currentNode()
        ctNode  = self.ctSelector.currentNode()
        if not mriNode:
            slicer.util.warningDisplay("Veuillez sélectionner un volume MRI T1.")
            return

        self.runAIButton.setEnabled(False)
        self.progressBar.setVisible(True)
        self.progressBar.setValue(0)
        self.statusLabel.setText("Détection en cours…")

        self.logic.runAIDetection(
            mriNode, ctNode,
            progressCallback    = self._onAIProgress,
            completionCallback  = self._onAICompleted,
            errorCallback       = self._onAIError
        )

    def _onAIProgress(self, percent, message=""):
        self.progressBar.setValue(percent)
        if message:
            self.statusLabel.setText(message)
        slicer.app.processEvents()

    def _onAICompleted(self, electrodes):
        """electrodes : list[dict] avec keys id, contacts, confidence, coords"""
        self.progressBar.setValue(100)
        self.runAIButton.setEnabled(True)
        self.statusLabel.setText(f"✔  {len(electrodes)} électrode(s) détectée(s)")

        self.electrodeManager.clearAll()
        for elec in electrodes:
            self.electrodeManager.addElectrode(elec)

        self._refreshElectrodeTable()
        if self.electrodeTable.rowCount > 0:
            self.electrodeTable.selectRow(0)
        self.progressBar.setVisible(False)

    def _onAIError(self, message):
        self.runAIButton.setEnabled(True)
        self.progressBar.setVisible(False)
        self.statusLabel.setText(f"✘  Erreur : {message}")
        self.statusLabel.setStyleSheet("color: #f44336; font-size: 12px;")

    # ─────────────────────────────────────────────────────────────────────────
    #  CALLBACKS – ELECTRODE
    # ─────────────────────────────────────────────────────────────────────────
    def _onBrainOpacityChanged(self, value):
        opacity = value / 100.0
        for node in slicer.util.getNodesByClass("vtkMRMLModelNode"):
            dn = node.GetDisplayNode()
            if dn:
                dn.SetOpacity(opacity)

    def _onElectrodeOpacityChanged(self, value):
        self.electrodeManager.setGlobalOpacity(value / 100.0)

    def _onShowLabelsChanged(self, state):
        self.electrodeManager.setLabelsVisible(state == qt.Qt.Checked)

    def _onShowContactsChanged(self, state):
        self.electrodeManager.setContactsNumberVisible(state == qt.Qt.Checked)

    def _onElectrodeSelected(self):
        rows = self.electrodeTable.selectedItems()
        if not rows:
            return
        row = self.electrodeTable.currentRow()
        elecId = self.electrodeTable.item(row, 1)
        if not elecId:
            return
        elecId = elecId.text()
        electrode = self.electrodeManager.getElectrode(elecId)
        if not electrode:
            return

        # Surbrillance dans la scène
        self.electrodeManager.highlightElectrode(elecId)

        # Afficher coordonnées du premier contact
        coords = electrode.get("coords", [])
        if coords:
            x, y, z = coords[0]
            self.coordsLabel.setText(
                f"Électrode {elecId}  —  "
                f"X: {x:.2f}  Y: {y:.2f}  Z: {z:.2f} mm\n"
                f"{len(coords)} contacts  |  Conf. {electrode.get('confidence', 0)*100:.0f}%"
            )
            # Centrer les vues 2D sur le premier contact
            self.logic.centerViewsOnPoint(x, y, z)

    def _onConfidenceThresholdChanged(self, value):
        threshold = value / 100.0
        self.electrodeManager.applyConfidenceThreshold(threshold)
        self._refreshElectrodeTable()

    def _onStatusFilterChanged(self):
        show = {
            "validated": self.showValidatedCheck.isChecked(),
            "average":   self.showAvgConfCheck.isChecked(),
            "low":       self.showLowConfCheck.isChecked(),
        }
        self.electrodeManager.applyStatusFilter(show)
        self._refreshElectrodeTable()

    # ─────────────────────────────────────────────────────────────────────────
    #  CALLBACKS – EXPORT
    # ─────────────────────────────────────────────────────────────────────────
    def _onExport(self):
        folder = qt.QFileDialog.getExistingDirectory(None, "Choisir le dossier d'export")
        if not folder:
            return

        fmt = "csv"
        if self.exportJSON.isChecked(): fmt = "json"
        if self.exportBIDS.isChecked(): fmt = "bids"
        if self.exportMRML.isChecked(): fmt = "mrml"

        options = {
            "format":           fmt,
            "include_conf":     self.includeConfCheck.isChecked(),
            "only_validated":   self.onlyValidatedCheck.isChecked(),
            "patient_id":       self.patientIDEdit.text,
        }

        try:
            manager = ExportManager(self.electrodeManager)
            path    = manager.export(folder, options)
            self.exportStatusLabel.setText(f"✔  Exporté : {os.path.basename(path)}")
            self.exportStatusLabel.setStyleSheet("color: #4caf50; font-size: 12px;")
        except Exception as e:
            self.exportStatusLabel.setText(f"✘  {str(e)}")
            self.exportStatusLabel.setStyleSheet("color: #f44336; font-size: 12px;")

    # ─────────────────────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _refreshElectrodeTable(self):
        # Sauvegarder la sélection actuelle
        selected_ids = []
        for item in self.electrodeTable.selectedItems():
            if item.column() == 1:
                selected_ids.append(item.text())

        electrodes = self.electrodeManager.getAllElectrodes()
        self.electrodeTable.blockSignals(True)
        self.electrodeTable.setRowCount(0)
        self._electrodeRows.clear()

        for elec in electrodes:
            row = self.electrodeTable.rowCount
            self.electrodeTable.insertRow(row)
            self._electrodeRows[elec["id"]] = row

            node = self.electrodeManager._nodes.get(elec["id"])
            is_displayed = False
            if node and node.GetDisplayNode():
                is_displayed = (node.GetDisplayNode().GetVisibility() == 1)

            text_color = qt.QBrush(qt.QColor(255, 255, 255) if is_displayed else qt.QColor(119, 119, 119))

            # Visibility checkbox
            visWidget = qt.QWidget()
            visLayout = qt.QHBoxLayout(visWidget)
            visLayout.setAlignment(qt.Qt.AlignCenter)
            visLayout.setContentsMargins(0, 0, 0, 0)
            visCheck = qt.QCheckBox()
            visCheck.setChecked(elec.get("visible", True))
            elecId = elec["id"]

            def make_on_vis_changed(eid, r, status):
                def on_changed(state):
                    self.electrodeManager.setVisibility(eid, state == qt.Qt.Checked)
                    # Mettre à jour la couleur du texte si la visibilité change
                    node_local = self.electrodeManager._nodes.get(eid)
                    is_disp = False
                    if node_local and node_local.GetDisplayNode():
                        is_disp = (node_local.GetDisplayNode().GetVisibility() == 1)
                    
                    color = qt.QColor(255, 255, 255) if is_disp else qt.QColor(119, 119, 119)
                    brush = qt.QBrush(color)
                    for col in range(1, 4):
                        item_local = self.electrodeTable.item(r, col)
                        if item_local:
                            item_local.setForeground(brush)
                            
                    status_w = self.electrodeTable.cellWidget(r, 4)
                    if status_w:
                        dot_label = status_w.findChild(qt.QLabel)
                        if dot_label:
                            dot_color = self._statusColor(status) if is_disp else "#777777"
                            dot_label.setStyleSheet(f"color: {dot_color}; font-size: 18px;")
                return on_changed

            visCheck.stateChanged.connect(make_on_vis_changed(elecId, row, elec.get("status", "low")))
            visLayout.addWidget(visCheck)
            self.electrodeTable.setCellWidget(row, 0, visWidget)

            # ID
            idItem = qt.QTableWidgetItem(elec["id"])
            idItem.setTextAlignment(qt.Qt.AlignCenter)
            idItem.setForeground(text_color)
            self.electrodeTable.setItem(row, 1, idItem)

            # Contacts
            cItem = qt.QTableWidgetItem(str(elec.get("contacts", 0)))
            cItem.setTextAlignment(qt.Qt.AlignCenter)
            cItem.setForeground(text_color)
            self.electrodeTable.setItem(row, 2, cItem)

            # Confidence
            confPct = int(elec.get("confidence", 0) * 100)
            confItem = qt.QTableWidgetItem(f"{confPct}%")
            confItem.setTextAlignment(qt.Qt.AlignCenter)
            confItem.setForeground(text_color)
            self.electrodeTable.setItem(row, 3, confItem)

            # Status (pastille colorée)
            statusWidget = qt.QWidget()
            statusLayout = qt.QHBoxLayout(statusWidget)
            statusLayout.setAlignment(qt.Qt.AlignCenter)
            statusLayout.setContentsMargins(0, 0, 0, 0)
            dot = qt.QLabel("●")
            dot_color = self._statusColor(elec.get("status", "low")) if is_displayed else "#777777"
            dot.setStyleSheet(f"color: {dot_color}; font-size: 18px;")
            statusLayout.addWidget(dot)
            self.electrodeTable.setCellWidget(row, 4, statusWidget)

            self.electrodeTable.setRowHeight(row, 32)
            
        self.electrodeTable.blockSignals(False)

        # Restaurer la sélection
        for r in range(self.electrodeTable.rowCount):
            item = self.electrodeTable.item(r, 1)
            if item and item.text() in selected_ids:
                self.electrodeTable.selectRow(r)

    def _statusColor(self, status):
        return {"validated": "#4caf50", "average": "#ff9800", "low": "#f44336"}.get(status, "#888")

    def _coloredCheckbox(self, label, color):
        cb = qt.QCheckBox(label)
        cb.setStyleSheet(f"color: {color}; font-size: 11px;")
        return cb

    def _btnStyle(self, bg):
        return f"""
            QPushButton {{
                background: {bg}; color: white;
                border-radius: 5px; padding: 8px;
            }}
            QPushButton:hover {{ background: #666; }}
        """

    def _onNodeAdded(self, caller, event):
        pass

    def cleanup(self):
        self.removeObservers()


# ─────────────────────────────────────────────────────────────────────────────
#  Logique métier
# ─────────────────────────────────────────────────────────────────────────────
class SEEGLocalizationLogic(ScriptedLoadableModuleLogic):

    def runAIDetection(self, mriNode, ctNode,
                       progressCallback=None,
                       completionCallback=None,
                       errorCallback=None):
        """Lance la détection IA en utilisant AIDetectionWorker."""
        try:
            worker = AIDetectionWorker(mriNode, ctNode)
            electrodes = worker.run(progressCallback=progressCallback)
            if completionCallback:
                completionCallback(electrodes)
        except Exception as e:
            if errorCallback:
                errorCallback(str(e))

    def centerViewsOnPoint(self, x, y, z):
        """Centre les vues 2D et 3D sur un point RAS donné."""
        layoutManager = slicer.app.layoutManager()
        
        # 2D views
        for name in ["Red", "Green", "Yellow"]:
            sliceWidget = layoutManager.sliceWidget(name)
            if sliceWidget:
                sliceNode = sliceWidget.mrmlSliceNode()
                sliceNode.JumpSliceByCentering(x, y, z)
                
        # 3D views
        for i in range(layoutManager.threeDViewCount):
            threeDWidget = layoutManager.threeDWidget(i)
            if threeDWidget:
                cameraNode = slicer.modules.cameras.logic().GetViewActiveCameraNode(threeDWidget.mrmlViewNode())
                if cameraNode:
                    camera = cameraNode.GetCamera()
                    focalPoint = np.array(camera.GetFocalPoint())
                    position = np.array(camera.GetPosition())
                    targetPoint = np.array([x, y, z])
                    translation = targetPoint - focalPoint
                    
                    camera.SetFocalPoint(targetPoint)
                    camera.SetPosition(position + translation)
