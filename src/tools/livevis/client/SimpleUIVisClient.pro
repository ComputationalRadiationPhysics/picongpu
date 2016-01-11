# Copyright 2013-2016 Benjamin Schneider, Axel Huebl, Rene Widera
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
#

#-------------------------------------------------
#
# Project created by QtCreator 2013-08-27T16:21:56
#
#-------------------------------------------------

QT       += core gui opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = SimpleUIVisClient
TEMPLATE = app
ONFIG += c++11
QMAKE_CXXFLAGS += -std=c++0x -DUSE_MJPEG=1


SOURCES += main.cpp\
        mainwindow.cpp \
    glwidget.cpp \
    simulationgallery.cpp \
    pictureflow.cpp \
    clickablelabel.cpp \
    udpquery.cpp \
    transferfunctions.cpp \
    verboselabel.cpp \
    camera.cpp

HEADERS  += mainwindow.h \
    glwidget.h \
    message_ids.hpp \
    simulationgallery.h \
    pictureflow.h \
    clickablelabel.h \
    udpquery.h \
    transferfunctions.h \
    verboselabel.h \
    camera.h \
    math_helper.h

FORMS    += mainwindow.ui \
    simulationgallery.ui


unix:!macx: LIBS += -L$$(RIVLIB_ROOT)/lib -lrivlib

INCLUDEPATH += $$(RIVLIB_ROOT)/include
DEPENDPATH += $$(RIVLIB_ROOT)/include

unix:!macx: LIBS += -L$$(VISLIB_ROOT)/lib/ -lvislibbase64

INCLUDEPATH += $$(VISLIB_ROOT)/base/include/vislib
DEPENDPATH += $$(VISLIB_ROOT)/base/include/vislib

unix:!macx: PRE_TARGETDEPS += $$(VISLIB_ROOT)/lib/libvislibbase64.a

unix:!macx: LIBS += -L$$(VISLIB_ROOT)/lib/ -lvislibsys64

INCLUDEPATH += $$(VISLIB_ROOT)/sys/include
DEPENDPATH += $$(VISLIB_ROOT)/sys/include

unix:!macx: PRE_TARGETDEPS += $$(VISLIB_ROOT)/lib/libvislibsys64.a

unix:!macx: LIBS += -L$$(THELIB_ROOT)/lib -lthelib

INCLUDEPATH += $$(THELIB_ROOT)/include
DEPENDPATH += $$(THELIB_ROOT)/include

LIBS += -lGLU

OTHER_FILES +=

RESOURCES += \
    icons.qrc
