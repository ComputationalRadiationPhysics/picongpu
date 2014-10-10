/********************************************************************************
** Form generated from reading UI file 'simulationgallery.ui'
**
** Created: Wed Jan 8 16:06:10 2014
**      by: Qt User Interface Compiler version 4.8.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SIMULATIONGALLERY_H
#define UI_SIMULATIONGALLERY_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHeaderView>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_SimulationGallery
{
public:

    void setupUi(QWidget *SimulationGallery)
    {
        if (SimulationGallery->objectName().isEmpty())
            SimulationGallery->setObjectName(QString::fromUtf8("SimulationGallery"));
        SimulationGallery->resize(767, 492);

        retranslateUi(SimulationGallery);

        QMetaObject::connectSlotsByName(SimulationGallery);
    } // setupUi

    void retranslateUi(QWidget *SimulationGallery)
    {
        SimulationGallery->setWindowTitle(QApplication::translate("SimulationGallery", "Form", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class SimulationGallery: public Ui_SimulationGallery {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SIMULATIONGALLERY_H
