import sys
import os
import copy
import pickle

import cv2
import numpy as np
import imutils
import math
import shutil
from PIL import Image

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QCoreApplication, QTimer, pyqtSlot, QAbstractTableModel, Qt
from PyQt5.uic import loadUi
import qdarkstyle

import torchvision
import torch
from torch import optim,nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms

print(torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DISEASE_CLASSES = {
                      0: 'nv, Melanocytic Nevi',
                      1: 'mel, Melanoma',
                      2: 'bkl, Benign Keratosis',
                      3: 'bcc, Basal Cell Carcinoma',
                      4: 'akiec, Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)',
                      5: 'vasc, Vascular skin lesion',
                      6: 'df, Dermatofibroma'

                  }


class Skin_Disease_Prediction(QMainWindow):
    def __init__(self):
        super(Skin_Disease_Prediction, self).__init__()
        loadUi('MainWindow.ui', self)
        os.system('cls')               
        
        self.train_algo_comboBox.activated.connect(self.Show_training_results)
        self.browse_pushButton.clicked.connect(self.BrowseFileDialog)
        self.Prediction_pushButton.clicked.connect(self.Classification_Function)

        self.qm = QMessageBox()
        
    @pyqtSlot()
    def Show_training_results(self):
        self.train_algo = str(self.train_algo_comboBox.currentText())
        
        if self.train_algo == 'DenseNet':
            img_1 = cv2.imread('./models/densenet/Model_evaluation.png')
            self.DisplayImage(img_1, 1)
            img_2 = cv2.imread('./models/densenet/conf_mat.png')
            self.DisplayImage(img_2, 2)
            img_3 = cv2.imread('./models/densenet/Fraction_classified_incorrectly.png')
            self.DisplayImage(img_3, 3)
            text = open('./models/densenet/classification_report.txt').read()
            self.plainTextEdit.setPlainText(text)


        elif self.train_algo == 'ResNet':
            img_1 = cv2.imread('./models/resnet/Model_evaluation.png')
            self.DisplayImage(img_1, 1)
            img_2 = cv2.imread('./models/resnet/conf_mat.png')
            self.DisplayImage(img_2, 2)
            img_3 = cv2.imread('./models/resnet/Fraction_classified_incorrectly.png')
            self.DisplayImage(img_3, 3)
            text = open('./models/resnet/classification_report.txt').read()
            self.plainTextEdit.setPlainText(text)


        elif self.train_algo == 'VGG':
            img_1 = cv2.imread('./models/vgg/Model_evaluation.png')
            self.DisplayImage(img_1, 1)
            img_2 = cv2.imread('./models/vgg/conf_mat.png')
            self.DisplayImage(img_2, 2)
            img_3 = cv2.imread('./models/vgg/Fraction_classified_incorrectly.png')
            self.DisplayImage(img_3, 3)
            text = open('./models/vgg/classification_report.txt').read()
            self.plainTextEdit.setPlainText(text)

    @pyqtSlot()
    def BrowseFileDialog(self):
        self.fname, filter = QFileDialog.getOpenFileName(self, 'Open image File', '.\\', "image Files (*.*)")
        if self.fname:
            self.LoadImageFunction(self.fname)
        else:
            print("No Valid File selected.")
            
    def LoadImageFunction(self, fname):
        self.image = cv2.imread(fname)
        self.DisplayImage(self.image, 0)

    def DisplayImage(self, img, window):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if (img.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImg = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)

        outImg = outImg.rgbSwapped()


        if window == 0:
            self.query_imglabel.setPixmap(QPixmap.fromImage(outImg))
            self.query_imglabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            self.query_imglabel.setScaledContents(True)
        elif window == 1:
            self.model_eval_imglabel.setPixmap(QPixmap.fromImage(outImg))
            self.model_eval_imglabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            self.model_eval_imglabel.setScaledContents(False)
        elif window == 2:
            self.confusion_matrix_imglabel.setPixmap(QPixmap.fromImage(outImg))
            self.confusion_matrix_imglabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            self.confusion_matrix_imglabel.setScaledContents(False)
        elif window == 3:
            self.fraction_incorrect_imglabel.setPixmap(QPixmap.fromImage(outImg))
            self.fraction_incorrect_imglabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            self.fraction_incorrect_imglabel.setScaledContents(False)

    
    @pyqtSlot()
    def Classification_Function(self):

        self.prediction_algo = str(self.prediction_algo_comboBox.currentText())
        
        if self.prediction_algo == 'DenseNet':
            self.DenseNet_Prediction()
        elif self.prediction_algo == 'ResNet':
            self.ResNet_Prediction()
        elif self.prediction_algo == 'VGG':
            self.VGG_Prediction()
        else:
            ret = self.qm.information(self,'Error !', 'No Algo Selected !\nPlease Select an algorithm', self.qm.Close)

    # ----------------------------------------------------------------------------------------------------------------------

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    # ----------------------------------------------------------------------------------------------------------------------

    def initialize_model(self, model_name, num_classes, feature_extract, use_pretrained=True):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0

        if model_name == "resnet":
            """ Resnet18, resnet34, resnet50, resnet101
            """
            model_ft = models.resnet50(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224


        elif model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224


        elif model_name == "densenet":
            """ Densenet121
            """
            model_ft = models.densenet121(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224


        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size

    # ----------------------------------------------------------------------------------------------------------------------

    def load_ckp(self, checkpoint_fpath, model, optimizer):
        checkpoint = torch.load(checkpoint_fpath)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        valid_loss_min = checkpoint['valid_loss_min']
        return model, optimizer, checkpoint['epoch'], valid_loss_min

    # ----------------------------------------------------------------------------------------------------------------------

    def Load_Training_Model(self, model_name):

        # model_name = model_name
        num_classes = 7
        feature_extract = False
        # Initialize the model for this run
        model_ft, input_size = self.initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
        # Define the device:
        device = torch.device('cuda:0')
        # Put the model on the device:
        model = model_ft.to(device)

        # we use Adam optimizer, use cross entropy loss as our loss function
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        best_model_path = ""

        if model_name == "densenet":
            ckp_path = "./models/densenet/current_checkpoint.pt"
            best_model_path = "./models/densenet/best_model.pt"
        elif model_name == "resnet":
            ckp_path = "./models/resnet/current_checkpoint.pt"
            best_model_path = "./models/resnet/best_model.pt"
        elif model_name == "vgg":
            ckp_path = "./models/vgg/current_checkpoint.pt"
            best_model_path = "./models/vgg/best_model.pt"


        # criterion = nn.CrossEntropyLoss().to(device)
        model, optimizer, start_epoch, valid_loss_min = self.load_ckp(best_model_path, model, optimizer)

        print("model = ", model)
        print("optimizer = ", optimizer)
        print("start_epoch = ", start_epoch)
        print("valid_loss_min = ", valid_loss_min)
        print("valid_loss_min = {:.6f}".format(valid_loss_min))

        # model.load_state_dict(torch.load('./best_model/best_model.pt'))
        model.eval()

        return model, input_size

    # ----------------------------------------------------------------------------------------------------------------------

    def Predict_Test_Image_File(self, model, model_name, input_size):

        print(self.fname)

        image = Image.open(self.fname)

        # Data from training step

        if model_name == "densenet":
            norm_mean = [0.7630344, 0.5456409, 0.57004315]
            norm_std = [0.14092779, 0.15261371, 0.16997148]
        elif model_name == "resnet":
            norm_mean = [0.7630327, 0.54564494, 0.5700448]
            norm_std = [0.1409281, 0.15261325, 0.16997032]
        elif model_name == "vgg":
            norm_mean = [0.7630364, 0.5456453, 0.57004297]
            norm_std = [0.1409282, 0.15261334, 0.16997081]


        # define the transformation of the test image.
        test_transforms = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(),
                                              transforms.Normalize(norm_mean, norm_std)])

        image_tensor = test_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(device)
        output = model(input)
        index = output.data.cpu().numpy().argmax()
        # return index

        disease_name = DISEASE_CLASSES[index]
        print(disease_name)
        self.prediction_result_label.setText(disease_name)


    # ----------------------------------------------------------------------
    
    def DenseNet_Prediction(self):
        model,input_size = self.Load_Training_Model('densenet')
        self.Predict_Test_Image_File(model, 'densenet', input_size)

# ----------------------------------------------------------------------

    def ResNet_Prediction(self):
        model,input_size = self.Load_Training_Model('resnet')
        self.Predict_Test_Image_File(model, 'resnet', input_size)

# ----------------------------------------------------------------------
    
    def VGG_Prediction(self):
        model,input_size = self.Load_Training_Model('vgg')
        self.Predict_Test_Image_File(model, 'vgg', input_size)

# ----------------------------------------------------------------------


''' ------------------------ MAIN Function ------------------------- '''
       
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = Skin_Disease_Prediction()
    window.show()
    sys.exit(app.exec_())
    