from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtCore import QDateTime, QRect, QTimer, Qt
from PyQt5.QtWidgets import QApplication, QCheckBox, QComboBox, QDateTimeEdit, QDial, QDialog, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLabel, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QRadioButton, QScrollArea, QScrollBar, QSizePolicy, QSlider, QSpinBox, QStyleFactory, QTabWidget, QTableWidget, QTextEdit, QVBoxLayout, QWidget
import colorThm as colorThm
import sys
from PyQt5.QtGui import QColor, QPalette

from lib_interface import QtGui, os, showDf, subprocess, shutil
import datamanip, arima , linearReg, json, file_manip


########### variables #################
STO = 1
DEP = 1
p,d,q,P,D,Q = 0,0,0,0,0,0
DMODEL = "additive"
FMODEL = None
FMODELLOAD = None
try:
    prefSaved = open("cached\prefeats.txt","r")
    line = prefSaved.readline()
    FTRS = line
except:
    FTRS = None
FOCUSED, TESTc, FEATURESc, TRAINc = file_manip.get_csv_config()

########### variables #################

class gui(QDialog):
    def __init__(self, parent=None):
        super(gui, self).__init__(parent)
        self.setWindowTitle("Walmart Sales Analytics")
        self.originalPalette = QApplication.palette()
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setMinimumSize(650,800)
        QApplication.setStyle('Fusion')
        try:
            prefSaved = open(file_manip.cachedPath()+"preference.txt","r")
            mode = prefSaved.readline()
            QApplication.setPalette(getattr(colorThm, mode)())
        except:
            print(1)
            QApplication.setPalette(colorThm.light_mode())

        ############# theme combobox #############
        winmode = QComboBox()
        winmode.addItems(['Light', 'Dark'])
        modename = QLabel("&Mode:")
        modename.setBuddy(winmode)
        winmode.activated[str].connect(self.changeMode)
        topLayout = QHBoxLayout()
        topLayout.addWidget(modename)
        topLayout.addWidget(winmode)
        topLayout.addStretch(1)
        ############# theme combobox #############

        
        self.createTopLeftGroupBox()
        self.createTopRightGroupBox()
        self.createBottomLeftTabWidget()
        self.createBottomRightGroupBox()


        mainLayout = QGridLayout()
        mainLayout.addLayout(topLayout, 0, 0, 1, 2)
        mainLayout.addWidget(self.topLeftGroupBox, 1, 0)
        mainLayout.addWidget(self.topRightGroupBox, 1, 1)
        mainLayout.addWidget(self.bottomLeftGroupBox, 2, 0)
        mainLayout.addWidget(self.bottomRightGroupBox, 2, 1)
        mainLayout.setRowStretch(1, 1)
        mainLayout.setRowStretch(2, 1)
        mainLayout.setColumnStretch(0, 1)
        mainLayout.setColumnStretch(1, 1)
        self.setLayout(mainLayout)


    def changeMode(self, styleName):
        mode = styleName.lower()+'_mode'
        QApplication.setPalette(getattr(colorThm, mode)())
        prefSaved = open("cached\preference.txt","w")
        prefSaved.write(mode)


    def modelSelect(self, model):
        global DMODEL 
        DMODEL = model
    
    def storeSelect(self, value):
        global STO 
        STO = int(value)
        print(str(STO) + ' ' + str(DEP))
 
    def deptSelect(self, value):
        global DEP 
        DEP = int(value)
        print(str(STO) + ' ' + str(DEP))

    def paramset(self):
        spinbx = self.sender()
        print()
        if (spinbx.objectName() == 'p'):
            global p
            p = int(spinbx.value())
            print('p - %d'%(p))
        elif (spinbx.objectName() == 'd'):
            global d
            d = int(spinbx.value())
            print('d - %d'%(d))
        elif (spinbx.objectName() == 'q'):
            global q
            q = int(spinbx.value())
            print('q - %d'%(q))
        elif (spinbx.objectName() == 'P'):
            global P
            P = int(spinbx.value())
        elif (spinbx.objectName() == 'D'):
            global D
            D = int(spinbx.value())
        elif (spinbx.objectName() == 'Q'):
            global Q
            Q = int(spinbx.value())

    def on_click(self):
        button = self.sender()
        global FMODEL
        try:
            train, test = datamanip.manip_data(STO, DEP, train_set = TRAINc, test_set = TESTc)
            if (button.objectName() =='dataframeBtn'):
                showDf(train, 'Store '+ str(STO) + ' Dept ' + str(DEP))
            elif (button.objectName() =='graphBtn'):
                datamanip.showgraph(train, 'Date', ['Weekly_Sales'], 'Date', 'Revenue', 'Store '+ str(STO) + ' Dept ' + str(DEP))
            elif (button.objectName() == 'heatmapBtn'):
                datamanip.corr_heatmap(train)
            elif (button.objectName() == 'decompBtn'):
                datamanip.show_decompose(train, model = DMODEL, focused = FOCUSED)
            elif (button.objectName() == 'arimaEvalBtn'):
                FMODEL = arima.check_prevision(train, p,d,q,P,D,Q, exog=FTRS, focused=FOCUSED)
            elif (button.objectName() == 'arimafrcBtn'):
                arima.plot_prevision(train,test, p,d,q,P,D,Q, exog=FTRS, focused=FOCUSED)
            elif (button.objectName() == "gridSearchBtn"):
                arima.gridSearch(train, exog=FTRS, focused=FOCUSED)
                result = ScrollMessageBox(file_manip.getAIC(), None)
                result.exec_()
            elif (button.objectName() == "ModelSaveBtn"):
                arima.modelSave(FMODEL, STO, DEP)
            elif (button.objectName() == 'linearfrcBtn'):
                linearReg.plotforecast(train, test)
            elif (button.objectName()== "ModelLoadBtn"):
                self.loadModel()
            else:
                QMessageBox.about(self, 'Command Error',"Unrecognised command") 
        except AttributeError:
            QMessageBox.about(self, 'Model failed to save',"Generate the Model by evaluating it first")  
        except MemoryError:
            QMessageBox.about(self, 'Graph failed to load',"Verify you csv_config.json file to make sure the feature to predict correspond to an existing feature") 
        except Exception:
            QMessageBox.about(self, 'Module failed to load', str(sys.exc_info())) 

    def updateData(self):
        path, filters = QFileDialog.getOpenFileName(self, 'Open CSV', os.getenv('PUBLIC')+'\Documents', 'CSV(*.csv)')
        if(path != ''):
            try:           
                file = os.path.basename(path)          
                os.unlink('./src/input/' + file)
            except:
                QMessageBox.about(self, 'Warning',"No previous entry of this file, corss check the filename with the csv_config.json")
            shutil.copy(path, './src/input/')
            print(file)   
        return
    
    def loadModel(self):
        import pathlib
        ptf = pathlib.Path(__file__).parent.absolute().parent.absolute().parent.absolute().parent.absolute()
        path, filters = QFileDialog.getOpenFileName(self, 'Open model json', str(ptf)+"\models", 'JSON(*.json)')
        global FMODELLOAD
        if(path != ''):
            try:           
                FMODELLOAD = path   
                print(FMODELLOAD)      
            except:
                QMessageBox.about(self, 'CSV Update Error',"Filename is of unknown format")   
        return


    def setFeature(self):
        global FTRS
        btn = self.sender()
        if btn.isChecked():
            FTRS = btn.text()
            featureSaved = open("cached\prefeats.txt","w")
            featureSaved.write(btn.text())
            featureSaved.close()
            print(btn.text())



    def createTopLeftGroupBox(self):
        self.topLeftGroupBox = QGroupBox("Select Store and Department")
        layout = QVBoxLayout()
        storecbx = QComboBox()
        storeId = []
        deptId = []
        for i in range(1, datamanip.csv_info(df=TRAINc, col="Store") + 1):
            storeId.append(str(i))
        for i in range(1, datamanip.csv_info(df=TRAINc, col="Dept") + 1):
            deptId.append(str(i))

        storecbx.addItems(storeId)
        storelbl = QLabel("&Store:")
        storelbl.setBuddy(storecbx)
        storecbx.activated[str].connect(self.storeSelect)

        deptcbx = QComboBox()
        deptcbx.addItems(deptId)
        deptlbl = QLabel("&Department:")
        deptlbl.setBuddy(storecbx)
        deptcbx.activated[str].connect(self.deptSelect)

        dataframeBtn = QPushButton('Show dataframe', self)
        dataframeBtn.setToolTip('Show the dfgui rendered pandas dataframe')
        dataframeBtn.setObjectName('dataframeBtn')
        dataframeBtn.clicked.connect(self.on_click)

        graphBtn = QPushButton('Show graph', self)
        graphBtn.setToolTip('Show the matplotlib rendered graph')
        graphBtn.setObjectName('graphBtn')
        graphBtn.clicked.connect(self.on_click)

        heatmapBtn = QPushButton('Show Heatmap', self)
        heatmapBtn.setToolTip('Show the matplotlib rendered Heatmap')
        heatmapBtn.setObjectName('heatmapBtn')
        heatmapBtn.clicked.connect(self.on_click)

        decompModel = QComboBox()
        decompModel.addItems(['additive', 'multiplicative'])
        decompModel.activated[str].connect(self.modelSelect)

        decompBtn = QPushButton('Seasonal Decomposition', self)
        decompBtn.setToolTip('Show Seasonal Decomposition of selected graph')
        decompBtn.setObjectName('decompBtn')
        decompBtn.clicked.connect(self.on_click)

        decomp = QHBoxLayout()
        decomp.addWidget(decompModel)
        decomp.addWidget(decompBtn)

        layout.addWidget(storelbl)
        layout.addWidget(storecbx)
        layout.addStretch(1)
        layout.addWidget(deptlbl)
        layout.addWidget(deptcbx)
        layout.addStretch(1)
        layout.addWidget(dataframeBtn)
        layout.addWidget(graphBtn)
        layout.addWidget(heatmapBtn)
        layout.addLayout(decomp)
        self.topLeftGroupBox.setLayout(layout)    



    def createTopRightGroupBox(self):
        self.topRightGroupBox = QGroupBox("CSV Info")

        csvInfo1lbl = QLabel("Current Train CSV showing data up to "+ datamanip.csv_info(df=TRAINc, info="date"))
        csvInfo2lbl = QLabel("Current Features CSV showing data up to "+ datamanip.csv_info(df=FEATURESc, info="date"))

        csvInfo3lbl = QLabel("Current Train CSV Full length: "+ str(datamanip.csv_info(df=TRAINc, info="len")))       
        csvInfo4lbl = QLabel("Current Features CSV Full length: "+ str(datamanip.csv_info(df=FEATURESc, info="len")))
        updateCsvBtn = QPushButton('Update CSV', self)
        updateCsvBtn.setToolTip('Upload a new CSV to the program')
        updateCsvBtn.clicked.connect(self.updateData)

        layout = QVBoxLayout()
        layout.addWidget(csvInfo1lbl)
        layout.addWidget(csvInfo2lbl)
        layout.addWidget(csvInfo3lbl)
        layout.addWidget(csvInfo4lbl)
        layout.addWidget(updateCsvBtn)
        layout.addStretch(1)
        self.topRightGroupBox.setLayout(layout)

    def createBottomLeftTabWidget(self):
        self.bottomLeftGroupBox = QGroupBox("Forcasters")
        layout = QGridLayout()
        tab = QTabWidget()
        tab.setSizePolicy(QSizePolicy.Preferred,
                QSizePolicy.Ignored)

        tab1 = QWidget()
        gridSearchBtn = QPushButton('Launch Grid Search', self)
        gridSearchBtn.setToolTip('Upload a new CSV to the program')
        gridSearchBtn.setObjectName('gridSearchBtn')
        gridSearchBtn.clicked.connect(self.on_click)

        ModelSaveBtn = QPushButton('Save Model', self)
        ModelSaveBtn.setToolTip('Save the fitted model for external use')
        ModelSaveBtn.setObjectName('ModelSaveBtn')
        ModelSaveBtn.clicked.connect(self.on_click)

        ModelLoadBtn = QPushButton('Load Model', self)
        ModelLoadBtn.setToolTip('Load an already fitted model to forecast')
        ModelLoadBtn.setObjectName('ModelLoadBtn')
        ModelLoadBtn.clicked.connect(self.on_click)

        arimaEvalBtn = QPushButton('Evaluate', self)
        arimaEvalBtn.setToolTip('Evaluate SARIMA Parameters')
        arimaEvalBtn.setObjectName('arimaEvalBtn')
        arimaEvalBtn.clicked.connect(self.on_click)

        arimafrcBtn = QPushButton('Forecast', self)
        arimafrcBtn.setToolTip('Forecast with SARIMA Parameters')
        arimafrcBtn.setObjectName('arimafrcBtn')
        arimafrcBtn.clicked.connect(self.on_click)

        p = QSpinBox(self.bottomLeftGroupBox)
        p.setMaximum(2)
        p.setValue(0)
        pLabel = QLabel("&AR")
        pLabel.setBuddy(p)
        p.setObjectName("p")
        p.valueChanged.connect(self.paramset)

        d = QSpinBox(self.bottomLeftGroupBox)
        d.setMaximum(2)
        d.setValue(0)
        dLabel = QLabel("&I")
        dLabel.setBuddy(d)
        d.setObjectName("d")
        d.valueChanged.connect(self.paramset)

        q = QSpinBox(self.bottomLeftGroupBox)
        q.setMaximum(2)
        q.setValue(0)
        qLabel = QLabel("&MA")
        qLabel.setBuddy(q)
        q.setObjectName("q")
        q.valueChanged.connect(self.paramset)

        P = QSpinBox(self.bottomLeftGroupBox)
        P.setMaximum(2)
        P.setValue(0)
        PLabel = QLabel("&S_AR")
        PLabel.setBuddy(P)
        P.setObjectName("P")
        P.valueChanged.connect(self.paramset)

        D = QSpinBox(self.bottomLeftGroupBox)
        D.setMaximum(2)
        D.setValue(0)
        DLabel = QLabel("&S_I")
        DLabel.setBuddy(D)
        D.setObjectName("D")
        D.valueChanged.connect(self.paramset)

        Q = QSpinBox(self.bottomLeftGroupBox)
        Q.setMaximum(2)
        Q.setValue(0)
        vQLabel = QLabel("&S_MA")
        vQLabel.setBuddy(Q)
        Q.setObjectName("Q")
        Q.valueChanged.connect(self.paramset)

        btnhbox = QHBoxLayout()
        btnhbox.addWidget(ModelSaveBtn)
        btnhbox.addWidget(ModelLoadBtn)

        line1hbox = QHBoxLayout()
        line2hbox = QHBoxLayout()
        tab1vbox = QVBoxLayout()
        line1hbox.setContentsMargins(0, 0, 0,0)
        line2hbox.setContentsMargins(0, 0, 0,0)
        line1hbox.addWidget(pLabel)
        line1hbox.addWidget(p)
        line1hbox.addWidget(dLabel)
        line1hbox.addWidget(d)
        line1hbox.addWidget(qLabel)
        line1hbox.addWidget(q)

        line2hbox.addWidget(PLabel)
        line2hbox.addWidget(P)
        line2hbox.addWidget(DLabel)
        line2hbox.addWidget(D)
        line2hbox.addWidget(vQLabel)
        line2hbox.addWidget(Q)
        
        tab1vbox.addLayout(line1hbox)
        tab1vbox.addLayout(line2hbox)
        tab1vbox.addWidget(gridSearchBtn)
        tab1vbox.addWidget(arimaEvalBtn)
        tab1vbox.addLayout(btnhbox)
        tab1vbox.addWidget(arimafrcBtn)

        tab1.setLayout(tab1vbox)
        

        tab2 = QWidget()
        linearfrcBtn = QPushButton('Forecast', self)
        linearfrcBtn.setToolTip('Launch linear regression forecasting')
        linearfrcBtn.setObjectName('linearfrcBtn')
        linearfrcBtn.clicked.connect(self.on_click)
        tab2vbox = QVBoxLayout()
        tab2vbox.addWidget(linearfrcBtn)
        tab2.setLayout(tab2vbox)

        tab.addTab(tab1, "&SARIMAX")
        tab.addTab(tab2, "&Linear Regression")
        layout.addWidget(tab)
        self.bottomLeftGroupBox.setLayout(layout)

    def createBottomRightGroupBox(self):
        self.bottomRightGroupBox = QGroupBox("Features")
        self.bottomRightGroupBox.setCheckable(True)
        self.bottomRightGroupBox.setChecked(True)



        layout = QGridLayout()
        ftrs = datamanip.returnFeatures()
        for f in ftrs:
            rdbtn = QRadioButton(f)
            rdbtn.toggled.connect(self.setFeature)
            layout.addWidget(rdbtn)
        self.bottomRightGroupBox.setLayout(layout)

class ScrollMessageBox(QMessageBox):
   def __init__(self, l, *args, **kwargs):
      QMessageBox.__init__(self, *args, **kwargs)
      scroll = QScrollArea(self)
      scroll.setWidgetResizable(True)
      self.content = QWidget()
      scroll.setWidget(self.content)
      lay = QVBoxLayout(self.content)
      for item in l:
         lay.addWidget(QLabel(item, self))
      self.layout().addWidget(scroll, 0, 0, 1, self.layout().columnCount())
      self.setStyleSheet("QScrollArea{min-width:800 px; min-height: 400px}")


if __name__ == '__main__':
    appctxt = ApplicationContext()
    interface = gui()
    interface.show()
    sys.exit(appctxt.app.exec_())
