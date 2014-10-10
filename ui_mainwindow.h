/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created: Fri Oct 10 13:47:17 2014
**      by: Qt User Interface Compiler version 4.8.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QComboBox>
#include <QtGui/QGridLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QMainWindow>
#include <QtGui/QPushButton>
#include <QtGui/QRadioButton>
#include <QtGui/QSlider>
#include <QtGui/QSpacerItem>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>
#include "clickablelabel.h"
#include "glwidget.h"
#include "verboselabel.h"

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    GLWidget *centralWidget;
    QHBoxLayout *horizontalLayout_3;
    QVBoxLayout *verticalLayout;
    QWidget *dataSidebar;
    QVBoxLayout *verticalLayout_2;
    QHBoxLayout *horizontalLayout;
    QVBoxLayout *verticalLayout_3;
    QLabel *label;
    QComboBox *cmbDatasourceA;
    QSpacerItem *verticalSpacer;
    QLabel *label_2;
    QComboBox *cmbDatasourceB;
    QSlider *sldSourceWeighting;
    QWidget *transferfuncSidebar;
    QVBoxLayout *verticalLayout_5;
    QHBoxLayout *horizontalLayout_4;
    QVBoxLayout *verticalLayout_6;
    QGroupBox *grpColorscaleA;
    QVBoxLayout *verticalLayout_9;
    QRadioButton *rdoRedGreenA;
    QRadioButton *rdoTempA;
    QRadioButton *rdo2HueA;
    QHBoxLayout *horizontalLayout_2;
    QSlider *sldOffsetXTFA;
    QSlider *sldSlopeTFA;
    QSlider *sldOffsetYTFA;
    VerboseLabel *lblTFA;
    QHBoxLayout *horizontalLayout_6;
    QVBoxLayout *verticalLayout_7;
    QGroupBox *grpColorscaleB;
    QVBoxLayout *verticalLayout_10;
    QRadioButton *rdoRedGreenB;
    QRadioButton *rdoTempB;
    QRadioButton *rdo2HueB;
    QHBoxLayout *horizontalLayout_5;
    QSlider *sldOffsetXTFB;
    QSlider *sldSlopeTFB;
    QSlider *sldOffsetYTFB;
    VerboseLabel *lblTFB;
    QWidget *settingsSidebar;
    QVBoxLayout *verticalLayout_8;
    QGroupBox *groupBox_3;
    QVBoxLayout *verticalLayout_4;
    QRadioButton *rdoAlphaBlending;
    QRadioButton *rdoMIP;
    QRadioButton *rdoIsoSurface;
    QLabel *label_9;
    QSlider *sldIsoValue;
    QLabel *label_5;
    ClickableLabel *lblBGColor;
    QSpacerItem *horizontalSpacer;
    QVBoxLayout *verticalLayout_11;
    QWidget *infoBar;
    QVBoxLayout *verticalLayout_12;
    QHBoxLayout *horizontalLayout_7;
    QPushButton *btnPlayPause;
    QPushButton *pushButton;
    QPushButton *btnWritePng;
    QLabel *label_3;
    QLabel *lblTimestep;
    QWidget *clipSidebar;
    QGridLayout *gridLayout;
    QLabel *label_6;
    QSlider *sldZMax;
    QSlider *sldZMin;
    QSlider *sldYMin;
    QLabel *label_4;
    QLabel *label_7;
    QSlider *sldYMax;
    QSlider *sldXMin;
    QSlider *sldXMax;
    QLabel *label_8;
    QSpacerItem *verticalSpacer_2;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(1680, 1050);
        MainWindow->setStyleSheet(QString::fromUtf8("/* QSlider Style Sheet */\n"
"\n"
"QSlider::handle:vertical {\n"
"        border: 1px solid grey;\n"
"        border-radius: 2px;\n"
"        background-color: #666666;\n"
"        margin: 0 -4px;\n"
"        height: 10px;\n"
"}\n"
"\n"
"QSlider::groove:vertical {\n"
"        width: 4px;\n"
"        border: 1px solid black;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"        border: 1px solid grey;\n"
"        border-radius: 2px;\n"
"        background-color: #666666;\n"
"        margin: -4px 0;\n"
"        width: 10px;\n"
"}\n"
"\n"
"QSlider::groove:horizontal {\n"
"        height: 4px;\n"
"        border: 1px solid black;\n"
"}\n"
"\n"
"QSlider::sub-page {\n"
"        background: #444444;\n"
"}\n"
"\n"
"QSlider::add-page {\n"
"        background: #336699;\n"
"}\n"
"\n"
"/* QRadioButton Style Sheet */\n"
"\n"
"QRadioButton::indicator {\n"
"        width: 16px;\n"
"        height: 16px;\n"
"        border: 2px outset grey;\n"
"        border-radius: 4px;\n"
"}\n"
"\n"
"QRadioButton::indicator::unchecked {\n"
"        background: #666666;\n"
"}\n"
"\n"
"QRadioButton::indicator::checked {\n"
"        background: #336699;\n"
"}"));
        centralWidget = new GLWidget(MainWindow);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        horizontalLayout_3 = new QHBoxLayout(centralWidget);
        horizontalLayout_3->setSpacing(0);
        horizontalLayout_3->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        horizontalLayout_3->setContentsMargins(0, 0, 0, 0);
        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(4);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(-1, 4, -1, 4);
        dataSidebar = new QWidget(centralWidget);
        dataSidebar->setObjectName(QString::fromUtf8("dataSidebar"));
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(dataSidebar->sizePolicy().hasHeightForWidth());
        dataSidebar->setSizePolicy(sizePolicy);
        dataSidebar->setMinimumSize(QSize(300, 0));
        dataSidebar->setStyleSheet(QString::fromUtf8("QWidget {\n"
"        background-color: #333333;\n"
"        color: #FFFFFF;\n"
"}\n"
"\n"
"#dataSidebar {\n"
"        border: 1px solid gray;\n"
"        border-left: 0px;\n"
"}"));
        verticalLayout_2 = new QVBoxLayout(dataSidebar);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(-1, -1, 6, -1);
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setSpacing(6);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        label = new QLabel(dataSidebar);
        label->setObjectName(QString::fromUtf8("label"));

        verticalLayout_3->addWidget(label);

        cmbDatasourceA = new QComboBox(dataSidebar);
        cmbDatasourceA->setObjectName(QString::fromUtf8("cmbDatasourceA"));

        verticalLayout_3->addWidget(cmbDatasourceA);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_3->addItem(verticalSpacer);

        label_2 = new QLabel(dataSidebar);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        verticalLayout_3->addWidget(label_2);

        cmbDatasourceB = new QComboBox(dataSidebar);
        cmbDatasourceB->setObjectName(QString::fromUtf8("cmbDatasourceB"));

        verticalLayout_3->addWidget(cmbDatasourceB);


        horizontalLayout->addLayout(verticalLayout_3);

        sldSourceWeighting = new QSlider(dataSidebar);
        sldSourceWeighting->setObjectName(QString::fromUtf8("sldSourceWeighting"));
        sldSourceWeighting->setStyleSheet(QString::fromUtf8(""));
        sldSourceWeighting->setMaximum(100);
        sldSourceWeighting->setValue(100);
        sldSourceWeighting->setOrientation(Qt::Vertical);

        horizontalLayout->addWidget(sldSourceWeighting);


        verticalLayout_2->addLayout(horizontalLayout);


        verticalLayout->addWidget(dataSidebar);

        transferfuncSidebar = new QWidget(centralWidget);
        transferfuncSidebar->setObjectName(QString::fromUtf8("transferfuncSidebar"));
        sizePolicy.setHeightForWidth(transferfuncSidebar->sizePolicy().hasHeightForWidth());
        transferfuncSidebar->setSizePolicy(sizePolicy);
        transferfuncSidebar->setMinimumSize(QSize(300, 0));
        transferfuncSidebar->setStyleSheet(QString::fromUtf8("QWidget {\n"
"        background-color: #333333;\n"
"        color: #FFFFFF;\n"
"}\n"
"\n"
"#transferfuncSidebar {\n"
"        border: 1px solid gray;\n"
"        border-left: 0px;\n"
"}"));
        verticalLayout_5 = new QVBoxLayout(transferfuncSidebar);
        verticalLayout_5->setSpacing(2);
        verticalLayout_5->setContentsMargins(11, 11, 11, 11);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        verticalLayout_5->setContentsMargins(-1, 1, 1, 1);
        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setSpacing(0);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        verticalLayout_6 = new QVBoxLayout();
        verticalLayout_6->setSpacing(6);
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        grpColorscaleA = new QGroupBox(transferfuncSidebar);
        grpColorscaleA->setObjectName(QString::fromUtf8("grpColorscaleA"));
        verticalLayout_9 = new QVBoxLayout(grpColorscaleA);
        verticalLayout_9->setSpacing(6);
        verticalLayout_9->setContentsMargins(11, 11, 11, 11);
        verticalLayout_9->setObjectName(QString::fromUtf8("verticalLayout_9"));
        rdoRedGreenA = new QRadioButton(grpColorscaleA);
        rdoRedGreenA->setObjectName(QString::fromUtf8("rdoRedGreenA"));

        verticalLayout_9->addWidget(rdoRedGreenA);

        rdoTempA = new QRadioButton(grpColorscaleA);
        rdoTempA->setObjectName(QString::fromUtf8("rdoTempA"));

        verticalLayout_9->addWidget(rdoTempA);

        rdo2HueA = new QRadioButton(grpColorscaleA);
        rdo2HueA->setObjectName(QString::fromUtf8("rdo2HueA"));

        verticalLayout_9->addWidget(rdo2HueA);


        verticalLayout_6->addWidget(grpColorscaleA);


        horizontalLayout_4->addLayout(verticalLayout_6);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(0);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        horizontalLayout_2->setSizeConstraint(QLayout::SetDefaultConstraint);
        horizontalLayout_2->setContentsMargins(-1, 4, -1, 4);
        sldOffsetXTFA = new QSlider(transferfuncSidebar);
        sldOffsetXTFA->setObjectName(QString::fromUtf8("sldOffsetXTFA"));
        sldOffsetXTFA->setMinimum(0);
        sldOffsetXTFA->setMaximum(100);
        sldOffsetXTFA->setValue(50);
        sldOffsetXTFA->setOrientation(Qt::Vertical);

        horizontalLayout_2->addWidget(sldOffsetXTFA);

        sldSlopeTFA = new QSlider(transferfuncSidebar);
        sldSlopeTFA->setObjectName(QString::fromUtf8("sldSlopeTFA"));
        sldSlopeTFA->setMaximum(100);
        sldSlopeTFA->setValue(50);
        sldSlopeTFA->setOrientation(Qt::Vertical);

        horizontalLayout_2->addWidget(sldSlopeTFA);

        sldOffsetYTFA = new QSlider(transferfuncSidebar);
        sldOffsetYTFA->setObjectName(QString::fromUtf8("sldOffsetYTFA"));
        sldOffsetYTFA->setMinimum(0);
        sldOffsetYTFA->setMaximum(100);
        sldOffsetYTFA->setValue(50);
        sldOffsetYTFA->setOrientation(Qt::Vertical);

        horizontalLayout_2->addWidget(sldOffsetYTFA);


        horizontalLayout_4->addLayout(horizontalLayout_2);

        lblTFA = new VerboseLabel(transferfuncSidebar);
        lblTFA->setObjectName(QString::fromUtf8("lblTFA"));
        sizePolicy.setHeightForWidth(lblTFA->sizePolicy().hasHeightForWidth());
        lblTFA->setSizePolicy(sizePolicy);
        lblTFA->setMinimumSize(QSize(40, 0));
        lblTFA->setScaledContents(true);

        horizontalLayout_4->addWidget(lblTFA);


        verticalLayout_5->addLayout(horizontalLayout_4);

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setSpacing(0);
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        verticalLayout_7 = new QVBoxLayout();
        verticalLayout_7->setSpacing(6);
        verticalLayout_7->setObjectName(QString::fromUtf8("verticalLayout_7"));
        grpColorscaleB = new QGroupBox(transferfuncSidebar);
        grpColorscaleB->setObjectName(QString::fromUtf8("grpColorscaleB"));
        verticalLayout_10 = new QVBoxLayout(grpColorscaleB);
        verticalLayout_10->setSpacing(6);
        verticalLayout_10->setContentsMargins(11, 11, 11, 11);
        verticalLayout_10->setObjectName(QString::fromUtf8("verticalLayout_10"));
        rdoRedGreenB = new QRadioButton(grpColorscaleB);
        rdoRedGreenB->setObjectName(QString::fromUtf8("rdoRedGreenB"));

        verticalLayout_10->addWidget(rdoRedGreenB);

        rdoTempB = new QRadioButton(grpColorscaleB);
        rdoTempB->setObjectName(QString::fromUtf8("rdoTempB"));

        verticalLayout_10->addWidget(rdoTempB);

        rdo2HueB = new QRadioButton(grpColorscaleB);
        rdo2HueB->setObjectName(QString::fromUtf8("rdo2HueB"));

        verticalLayout_10->addWidget(rdo2HueB);


        verticalLayout_7->addWidget(grpColorscaleB);


        horizontalLayout_6->addLayout(verticalLayout_7);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setSpacing(0);
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        horizontalLayout_5->setContentsMargins(-1, 4, -1, 4);
        sldOffsetXTFB = new QSlider(transferfuncSidebar);
        sldOffsetXTFB->setObjectName(QString::fromUtf8("sldOffsetXTFB"));
        sldOffsetXTFB->setMinimum(0);
        sldOffsetXTFB->setMaximum(100);
        sldOffsetXTFB->setValue(50);
        sldOffsetXTFB->setOrientation(Qt::Vertical);

        horizontalLayout_5->addWidget(sldOffsetXTFB);

        sldSlopeTFB = new QSlider(transferfuncSidebar);
        sldSlopeTFB->setObjectName(QString::fromUtf8("sldSlopeTFB"));
        sldSlopeTFB->setMaximum(100);
        sldSlopeTFB->setValue(50);
        sldSlopeTFB->setOrientation(Qt::Vertical);

        horizontalLayout_5->addWidget(sldSlopeTFB);

        sldOffsetYTFB = new QSlider(transferfuncSidebar);
        sldOffsetYTFB->setObjectName(QString::fromUtf8("sldOffsetYTFB"));
        sldOffsetYTFB->setMinimum(0);
        sldOffsetYTFB->setMaximum(100);
        sldOffsetYTFB->setValue(50);
        sldOffsetYTFB->setOrientation(Qt::Vertical);

        horizontalLayout_5->addWidget(sldOffsetYTFB);


        horizontalLayout_6->addLayout(horizontalLayout_5);

        lblTFB = new VerboseLabel(transferfuncSidebar);
        lblTFB->setObjectName(QString::fromUtf8("lblTFB"));
        sizePolicy.setHeightForWidth(lblTFB->sizePolicy().hasHeightForWidth());
        lblTFB->setSizePolicy(sizePolicy);
        lblTFB->setMinimumSize(QSize(40, 0));
        lblTFB->setScaledContents(true);

        horizontalLayout_6->addWidget(lblTFB);


        verticalLayout_5->addLayout(horizontalLayout_6);


        verticalLayout->addWidget(transferfuncSidebar);

        settingsSidebar = new QWidget(centralWidget);
        settingsSidebar->setObjectName(QString::fromUtf8("settingsSidebar"));
        sizePolicy.setHeightForWidth(settingsSidebar->sizePolicy().hasHeightForWidth());
        settingsSidebar->setSizePolicy(sizePolicy);
        settingsSidebar->setMinimumSize(QSize(300, 0));
        settingsSidebar->setStyleSheet(QString::fromUtf8("QWidget {\n"
"        background-color: #333333;\n"
"        color: #FFFFFF;\n"
"}\n"
"\n"
"#settingsSidebar {\n"
"        border: 1px solid gray;\n"
"        border-left: 0px;\n"
"}"));
        verticalLayout_8 = new QVBoxLayout(settingsSidebar);
        verticalLayout_8->setSpacing(6);
        verticalLayout_8->setContentsMargins(11, 11, 11, 11);
        verticalLayout_8->setObjectName(QString::fromUtf8("verticalLayout_8"));
        verticalLayout_8->setContentsMargins(-1, -1, 1, -1);
        groupBox_3 = new QGroupBox(settingsSidebar);
        groupBox_3->setObjectName(QString::fromUtf8("groupBox_3"));
        verticalLayout_4 = new QVBoxLayout(groupBox_3);
        verticalLayout_4->setSpacing(6);
        verticalLayout_4->setContentsMargins(11, 11, 11, 11);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        rdoAlphaBlending = new QRadioButton(groupBox_3);
        rdoAlphaBlending->setObjectName(QString::fromUtf8("rdoAlphaBlending"));
        rdoAlphaBlending->setLayoutDirection(Qt::RightToLeft);

        verticalLayout_4->addWidget(rdoAlphaBlending);

        rdoMIP = new QRadioButton(groupBox_3);
        rdoMIP->setObjectName(QString::fromUtf8("rdoMIP"));
        rdoMIP->setLayoutDirection(Qt::RightToLeft);

        verticalLayout_4->addWidget(rdoMIP);

        rdoIsoSurface = new QRadioButton(groupBox_3);
        rdoIsoSurface->setObjectName(QString::fromUtf8("rdoIsoSurface"));
        rdoIsoSurface->setLayoutDirection(Qt::RightToLeft);

        verticalLayout_4->addWidget(rdoIsoSurface);

        label_9 = new QLabel(groupBox_3);
        label_9->setObjectName(QString::fromUtf8("label_9"));

        verticalLayout_4->addWidget(label_9);

        sldIsoValue = new QSlider(groupBox_3);
        sldIsoValue->setObjectName(QString::fromUtf8("sldIsoValue"));
        sldIsoValue->setMaximum(100);
        sldIsoValue->setValue(50);
        sldIsoValue->setOrientation(Qt::Horizontal);

        verticalLayout_4->addWidget(sldIsoValue);


        verticalLayout_8->addWidget(groupBox_3);

        label_5 = new QLabel(settingsSidebar);
        label_5->setObjectName(QString::fromUtf8("label_5"));
        label_5->setAlignment(Qt::AlignBottom|Qt::AlignLeading|Qt::AlignLeft);

        verticalLayout_8->addWidget(label_5);

        lblBGColor = new ClickableLabel(settingsSidebar);
        lblBGColor->setObjectName(QString::fromUtf8("lblBGColor"));
        lblBGColor->setStyleSheet(QString::fromUtf8("QLabel {\n"
"        background-color: black;\n"
"}"));
        lblBGColor->setTextFormat(Qt::AutoText);
        lblBGColor->setScaledContents(true);
        lblBGColor->setAlignment(Qt::AlignCenter);

        verticalLayout_8->addWidget(lblBGColor);


        verticalLayout->addWidget(settingsSidebar);

        verticalLayout->setStretch(0, 1);
        verticalLayout->setStretch(1, 1);
        verticalLayout->setStretch(2, 1);

        horizontalLayout_3->addLayout(verticalLayout);

        horizontalSpacer = new QSpacerItem(639, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_3->addItem(horizontalSpacer);

        verticalLayout_11 = new QVBoxLayout();
        verticalLayout_11->setSpacing(4);
        verticalLayout_11->setObjectName(QString::fromUtf8("verticalLayout_11"));
        verticalLayout_11->setContentsMargins(-1, 4, 0, -1);
        infoBar = new QWidget(centralWidget);
        infoBar->setObjectName(QString::fromUtf8("infoBar"));
        QSizePolicy sizePolicy1(QSizePolicy::Minimum, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(infoBar->sizePolicy().hasHeightForWidth());
        infoBar->setSizePolicy(sizePolicy1);
        infoBar->setMinimumSize(QSize(0, 40));
        infoBar->setMaximumSize(QSize(16777215, 40));
        infoBar->setStyleSheet(QString::fromUtf8("QWidget {\n"
"        background-color: #333333;\n"
"        color: #FFFFFF;\n"
"}\n"
"\n"
"QWidget#infoBar {\n"
"        border: 1px solid gray;\n"
"        border-right: 0px;\n"
"}"));
        verticalLayout_12 = new QVBoxLayout(infoBar);
        verticalLayout_12->setSpacing(0);
        verticalLayout_12->setContentsMargins(11, 11, 11, 11);
        verticalLayout_12->setObjectName(QString::fromUtf8("verticalLayout_12"));
        verticalLayout_12->setContentsMargins(1, 1, 0, 1);
        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setSpacing(6);
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));
        horizontalLayout_7->setContentsMargins(6, -1, -1, -1);
        btnPlayPause = new QPushButton(infoBar);
        btnPlayPause->setObjectName(QString::fromUtf8("btnPlayPause"));
        btnPlayPause->setMinimumSize(QSize(30, 30));
        btnPlayPause->setMaximumSize(QSize(30, 30));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/icons/play_button.png"), QSize(), QIcon::Normal, QIcon::On);
        btnPlayPause->setIcon(icon);
        btnPlayPause->setFlat(false);

        horizontalLayout_7->addWidget(btnPlayPause);

        pushButton = new QPushButton(infoBar);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));
        pushButton->setMinimumSize(QSize(30, 30));
        pushButton->setMaximumSize(QSize(30, 30));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/icons/home_button.png"), QSize(), QIcon::Normal, QIcon::Off);
        pushButton->setIcon(icon1);

        horizontalLayout_7->addWidget(pushButton);

        btnWritePng = new QPushButton(infoBar);
        btnWritePng->setObjectName(QString::fromUtf8("btnWritePng"));
        btnWritePng->setMinimumSize(QSize(30, 30));
        btnWritePng->setMaximumSize(QSize(30, 30));

        horizontalLayout_7->addWidget(btnWritePng);

        label_3 = new QLabel(infoBar);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        horizontalLayout_7->addWidget(label_3);

        lblTimestep = new QLabel(infoBar);
        lblTimestep->setObjectName(QString::fromUtf8("lblTimestep"));
        lblTimestep->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        lblTimestep->setMargin(8);

        horizontalLayout_7->addWidget(lblTimestep);


        verticalLayout_12->addLayout(horizontalLayout_7);


        verticalLayout_11->addWidget(infoBar);

        clipSidebar = new QWidget(centralWidget);
        clipSidebar->setObjectName(QString::fromUtf8("clipSidebar"));
        QSizePolicy sizePolicy2(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(clipSidebar->sizePolicy().hasHeightForWidth());
        clipSidebar->setSizePolicy(sizePolicy2);
        clipSidebar->setMinimumSize(QSize(0, 0));
        clipSidebar->setStyleSheet(QString::fromUtf8("QWidget {\n"
"        background-color: #333333;\n"
"        color: #FFFFFF;\n"
"}\n"
"\n"
"#clipSidebar {\n"
"        border: 1px solid gray;\n"
"        border-right: 0px;\n"
"}"));
        gridLayout = new QGridLayout(clipSidebar);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        label_6 = new QLabel(clipSidebar);
        label_6->setObjectName(QString::fromUtf8("label_6"));
        label_6->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout->addWidget(label_6, 1, 2, 1, 1);

        sldZMax = new QSlider(clipSidebar);
        sldZMax->setObjectName(QString::fromUtf8("sldZMax"));
        sizePolicy2.setHeightForWidth(sldZMax->sizePolicy().hasHeightForWidth());
        sldZMax->setSizePolicy(sizePolicy2);
        sldZMax->setMaximum(100);
        sldZMax->setValue(100);
        sldZMax->setOrientation(Qt::Vertical);

        gridLayout->addWidget(sldZMax, 1, 3, 1, 1);

        sldZMin = new QSlider(clipSidebar);
        sldZMin->setObjectName(QString::fromUtf8("sldZMin"));
        sizePolicy2.setHeightForWidth(sldZMin->sizePolicy().hasHeightForWidth());
        sldZMin->setSizePolicy(sizePolicy2);
        sldZMin->setMaximum(100);
        sldZMin->setOrientation(Qt::Vertical);

        gridLayout->addWidget(sldZMin, 3, 3, 1, 1);

        sldYMin = new QSlider(clipSidebar);
        sldYMin->setObjectName(QString::fromUtf8("sldYMin"));
        sizePolicy2.setHeightForWidth(sldYMin->sizePolicy().hasHeightForWidth());
        sldYMin->setSizePolicy(sizePolicy2);
        sldYMin->setMaximum(100);
        sldYMin->setOrientation(Qt::Vertical);

        gridLayout->addWidget(sldYMin, 3, 1, 1, 1);

        label_4 = new QLabel(clipSidebar);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        label_4->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout->addWidget(label_4, 1, 0, 1, 1);

        label_7 = new QLabel(clipSidebar);
        label_7->setObjectName(QString::fromUtf8("label_7"));
        label_7->setAlignment(Qt::AlignHCenter|Qt::AlignTop);

        gridLayout->addWidget(label_7, 3, 0, 1, 1);

        sldYMax = new QSlider(clipSidebar);
        sldYMax->setObjectName(QString::fromUtf8("sldYMax"));
        sizePolicy2.setHeightForWidth(sldYMax->sizePolicy().hasHeightForWidth());
        sldYMax->setSizePolicy(sizePolicy2);
        sldYMax->setMaximum(100);
        sldYMax->setValue(100);
        sldYMax->setOrientation(Qt::Vertical);

        gridLayout->addWidget(sldYMax, 1, 1, 1, 1);

        sldXMin = new QSlider(clipSidebar);
        sldXMin->setObjectName(QString::fromUtf8("sldXMin"));
        sizePolicy2.setHeightForWidth(sldXMin->sizePolicy().hasHeightForWidth());
        sldXMin->setSizePolicy(sizePolicy2);
        sldXMin->setMaximum(100);
        sldXMin->setOrientation(Qt::Horizontal);

        gridLayout->addWidget(sldXMin, 2, 0, 1, 1);

        sldXMax = new QSlider(clipSidebar);
        sldXMax->setObjectName(QString::fromUtf8("sldXMax"));
        sizePolicy2.setHeightForWidth(sldXMax->sizePolicy().hasHeightForWidth());
        sldXMax->setSizePolicy(sizePolicy2);
        sldXMax->setMaximum(100);
        sldXMax->setValue(100);
        sldXMax->setOrientation(Qt::Horizontal);

        gridLayout->addWidget(sldXMax, 2, 2, 1, 1);

        label_8 = new QLabel(clipSidebar);
        label_8->setObjectName(QString::fromUtf8("label_8"));

        gridLayout->addWidget(label_8, 0, 0, 1, 1);


        verticalLayout_11->addWidget(clipSidebar);

        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_11->addItem(verticalSpacer_2);


        horizontalLayout_3->addLayout(verticalLayout_11);

        MainWindow->setCentralWidget(centralWidget);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("MainWindow", "Datasource A", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("MainWindow", "Datasource B", 0, QApplication::UnicodeUTF8));
        grpColorscaleA->setTitle(QApplication::translate("MainWindow", "No Datasource", 0, QApplication::UnicodeUTF8));
        rdoRedGreenA->setText(QApplication::translate("MainWindow", "Red-Green", 0, QApplication::UnicodeUTF8));
        rdoTempA->setText(QApplication::translate("MainWindow", "Temperature", 0, QApplication::UnicodeUTF8));
        rdo2HueA->setText(QApplication::translate("MainWindow", "2-Hue", 0, QApplication::UnicodeUTF8));
        lblTFA->setText(QString());
        grpColorscaleB->setTitle(QApplication::translate("MainWindow", "No Datasource", 0, QApplication::UnicodeUTF8));
        rdoRedGreenB->setText(QApplication::translate("MainWindow", "Red-Green", 0, QApplication::UnicodeUTF8));
        rdoTempB->setText(QApplication::translate("MainWindow", "Temperature", 0, QApplication::UnicodeUTF8));
        rdo2HueB->setText(QApplication::translate("MainWindow", "2-Hue", 0, QApplication::UnicodeUTF8));
        lblTFB->setText(QString());
        groupBox_3->setTitle(QApplication::translate("MainWindow", "Compositing Mode", 0, QApplication::UnicodeUTF8));
        rdoAlphaBlending->setText(QApplication::translate("MainWindow", "Alpha Blending  ", 0, QApplication::UnicodeUTF8));
        rdoMIP->setText(QApplication::translate("MainWindow", "Maximum Intensity Projection  ", 0, QApplication::UnicodeUTF8));
        rdoIsoSurface->setText(QApplication::translate("MainWindow", "Iso Surface  ", 0, QApplication::UnicodeUTF8));
        label_9->setText(QApplication::translate("MainWindow", "Iso Value", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("MainWindow", "Backgroundcolor", 0, QApplication::UnicodeUTF8));
        btnPlayPause->setText(QString());
        pushButton->setText(QString());
        btnWritePng->setText(QApplication::translate("MainWindow", "Off", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("MainWindow", "Timestep:", 0, QApplication::UnicodeUTF8));
        lblTimestep->setText(QString());
        label_6->setText(QApplication::translate("MainWindow", "Z", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("MainWindow", "Y", 0, QApplication::UnicodeUTF8));
        label_7->setText(QApplication::translate("MainWindow", "X", 0, QApplication::UnicodeUTF8));
        label_8->setText(QApplication::translate("MainWindow", "Clipping", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
