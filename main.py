import sys

import matplotlib.figure
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as sm
from PyQt6 import QtWidgets
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import ListedColormap
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from MainWindow import Ui_MainWindow

models = [LogisticRegression(), SVC(), RandomForestClassifier(), KNeighborsClassifier(), GaussianNB(), MLPClassifier()]
kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.prev_batch_size = 0.8
        self.first_plot = False
        self.std_data = None
        self.ax = None
        self.y_test = None
        self.y_train = None
        self.x_test = None
        self.x_train = None
        self.markers = None
        self.cmap = None
        self.kernel = 0
        self.classifier = 0
        self.col_label = None
        self.col_end = None
        self.col_start = None
        self.setupUi(self)
        self.file_name = None
        self.batch_size = 0.8
        self.psh_bt_browse.setText("")
        search_icon = QIcon("search.svg")
        self.psh_bt_browse.setIcon(search_icon)
        self.psh_bt_browse.setIconSize(QSize(25, 25))
        self.psh_bt_browse.setFixedSize(38, 38)
        # self.ln_edt_custom_training.setMaximumSize(100)
        self.psh_bt_browse.clicked.connect(self.psh_bt_browse_clicked)
        self.psh_bt_set_col.clicked.connect(self.psh_bt_set_col_clicked)
        self.psh_bt_set_batch.clicked.connect(self.psh_bt_set_batch_clicked)
        self.cmb_bx_classifier.currentIndexChanged.connect(self.cmb_bx_classifier_changed)
        self.cmb_bx_kernel.currentIndexChanged.connect(self.cmb_bx_kernel_changed)
        self.chk_bx_data_std.checkStateChanged.connect(self.chk_bx_data_std_changed)
        self.chk_bx_custom_training.checkStateChanged.connect(self.chk_bx_custom_training_changed)

        self.figure_model_visualizer = matplotlib.figure.Figure()
        self.canvas_model_visualizer = FigureCanvas(self.figure_model_visualizer)
        self.vt_lt_model_visualizer.addWidget(self.canvas_model_visualizer)

        self.figure_confusion_matrix = matplotlib.figure.Figure()
        self.canvas_confusion_matrix = FigureCanvas(self.figure_confusion_matrix)
        self.vt_lt_confusion_matrix.addWidget(self.canvas_confusion_matrix)

        self.figure_roc_curve = matplotlib.figure.Figure()
        self.canvas_roc_curve = FigureCanvas(self.figure_roc_curve)
        self.vt_lt_roc_curve.addWidget(self.canvas_roc_curve)

    def chk_bx_custom_training_changed(self):
        if not self.chk_bx_custom_training.isChecked():
            self.batch_size = 0.8
        else:
            self.batch_size = self.prev_batch_size
        self.plot_dataset()

    def chk_bx_data_std_changed(self):
        self.std_data = self.chk_bx_data_std.isChecked()
        print(self.std_data)
        self.plot_dataset() if self.first_plot else None

    def cmb_bx_classifier_changed(self):
        self.classifier = int(self.cmb_bx_classifier.currentIndex())
        self.plot_dataset()

    def cmb_bx_kernel_changed(self):
        self.kernel = int(self.cmb_bx_kernel.currentIndex())
        self.plot_dataset()

    def psh_bt_set_col_clicked(self):
        self.col_start = int(self.ln_edt_start.text()) if self.ln_edt_start.text() != "" else None
        self.col_end = int(self.ln_edt_end.text()) if self.ln_edt_end.text() != "" else None
        self.col_label = int(self.ln_edt_sample_lables.text()) if self.ln_edt_sample_lables.text() != "" else None
        self.plot_dataset()

    def psh_bt_browse_clicked(self):
        self.file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)")
        self.ln_edt_browse.setText(self.file_name)
        self.plot_dataset()

    def psh_bt_set_batch_clicked(self):
        self.batch_size = float(
            self.ln_edt_custom_training.text()) / 100 if self.ln_edt_custom_training.text() != "" else 0.8
        self.prev_batch_size = self.batch_size
        self.ln_edt_custom_training.setText(str(self.batch_size * 100))
        print(self.chk_bx_custom_training.isChecked())
        if self.chk_bx_custom_training.isChecked():
            print("checked custom training")
            self.plot_dataset()

    def load_dataset(self):
        df = pd.read_csv(self.file_name)
        x = df[df.columns[self.col_start - 1:self.col_end]]
        y_categories = df[df.columns[self.col_label - 1]]
        y = np.int32(y_categories)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, train_size=self.batch_size, random_state=0
        )

        self.markers = ("s", "x", "o", "^", "v")
        colors = ("red", "blue", "lightgreen", "gray", "cyan")
        self.cmap = ListedColormap(colors[:len(np.unique(y))])

    def plot_dataset(self):
        if self.col_label is None or self.col_end is None or self.col_start is None:
            return

        self.load_dataset()

        x_test_std = None

        # important step before we apply PCA, cause data needs to have mean = 0 and variance = 1
        if self.std_data:
            sc = StandardScaler()
            self.x_train = sc.fit_transform(self.x_train)

            x_test_std = sc.transform(self.x_test)

        model = models[self.cmb_bx_classifier.currentIndex()]
        kernel = kernels[self.cmb_bx_kernel.currentIndex()]

        kernel_pca = KernelPCA(n_components=2, kernel=kernel)
        x_train_pca = kernel_pca.fit_transform(self.x_train)
        model.fit(x_train_pca, self.y_train)

        # x_test_pca contains two data array containing two Principal Components of the given dataset

        if self.std_data:
            x_test_pca = kernel_pca.transform(x_test_std)
        else:
            x_test_pca = kernel_pca.transform(self.x_test)

        y_pred = model.predict(x_test_pca)

        accuracy = sm.accuracy_score(self.y_test, y_pred)
        accuracy = str(np.round(accuracy, decimals=3))
        precision = "NA"
        recall = "NA"
        f1_score = "NA"
        if len(np.unique(self.y_test)) == 2:
            precision = sm.precision_score(self.y_test, y_pred)
            precision = str(np.round(precision, decimals=3))
            recall = sm.recall_score(self.y_test, y_pred)
            recall = str(np.round(recall, decimals=3))
            f1_score = sm.f1_score(self.y_test, y_pred)
            f1_score = str(np.round(f1_score, decimals=3))
        self.txt_lb_precision.setText(precision)
        self.txt_lb_recall.setText(recall)
        self.txt_lb_accuracy.setText(accuracy)
        self.txt_lbl_f1_score.setText(f1_score)

        self.figure_model_visualizer.clear()

        ax = self.figure_model_visualizer.add_subplot(111)
        self.plot_decision_regions(x_train_pca, self.y_train, model, ax)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend(loc="lower left")
        self.canvas_model_visualizer.draw()

        self.figure_confusion_matrix.clear()
        y_predict = y_pred.astype(int)
        y_predict = y_predict.reshape(-1, 1)
        y_train = self.y_train.astype(int)
        y_train = y_train.reshape(-1, 1)

        if y_train.shape != y_predict.shape:
            min_len = min(len(y_train), len(y_predict))
            y_train = y_train[:min_len]
            y_predict = y_predict[:min_len]
            yy_train_flat = y_train.flatten()
            yy_predict_flat = y_predict.flatten()
            cm = confusion_matrix(yy_train_flat, yy_predict_flat)
            ax = self.figure_confusion_matrix.add_subplot(111)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(yy_train_flat),
                        yticklabels=np.unique(yy_train_flat), ax=ax)
            ax.set_xlabel('Predicted Labels')
            ax.set_ylabel('True Labels')

        self.canvas_confusion_matrix.draw()

        self.figure_roc_curve.clear()
        if len(np.unique(self.y_test)) == 2 and self.cmb_bx_classifier.currentIndex() != 1:
            y_prob = model.predict_proba(x_test_pca)[:, 1]
            fpr, tpr, thresholds = roc_curve(self.y_test, y_prob)

            auc = roc_auc_score(self.y_test, y_prob)

            ax = self.figure_roc_curve.add_subplot(111)

            ax.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % auc)
            ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")

        self.first_plot = True
        self.canvas_roc_curve.draw()

    def plot_decision_regions(self, x, y, classifier, ax, resolution=0.02):
        x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution, dtype=np.float32),
                               np.arange(x2_min, x2_max, resolution, dtype=np.float32))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)

        ax.contourf(xx1, xx2, Z, alpha=0.4, cmap=self.cmap)
        ax.set_xlim(xx1.min(), xx1.max())
        ax.set_ylim(xx2.min(), xx2.max())
        for idx, cl in enumerate(np.unique(y)):
            ax.scatter(x=x[y == cl, 0], y=x[y == cl, 1], alpha=0.8, c=self.cmap(idx), marker=self.markers[idx],
                       label=cl)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()
    app.exec()
