/****************************************************************************
** Meta object code from reading C++ file 'mainwindow.h'
**
** Created: Fri Oct 10 13:47:30 2014
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "mainwindow.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'mainwindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_MainWindow[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
      46,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       5,       // signalCount

 // signals: signature, parameters, type, tag, flags
      17,   12,   11,   11, 0x05,
      50,   43,   11,   11, 0x05,
      82,   11,   11,   11, 0x05,
     113,   11,   11,   11, 0x05,
     144,   11,   11,   11, 0x05,

 // slots: signature, parameters, type, tag, flags
     175,   11,   11,   11, 0x0a,
     196,   11,   11,   11, 0x0a,
     225,   11,   11,   11, 0x0a,
     250,   11,   11,   11, 0x0a,
     267,   11,   11,   11, 0x0a,
     284,   11,   11,   11, 0x0a,
     296,   11,   11,   11, 0x0a,
     314,   11,   11,   11, 0x0a,
     341,   11,   11,   11, 0x0a,
     369,  365,   11,   11, 0x0a,
     391,   43,   11,   11, 0x0a,
     420,   12,   11,   11, 0x0a,
     456,  443,   11,   11, 0x0a,
     491,   11,   11,   11, 0x0a,
     519,   11,   11,   11, 0x0a,
     547,   11,   11,   11, 0x0a,
     562,   11,   11,   11, 0x0a,
     580,   11,   11,   11, 0x0a,
     604,  599,   11,   11, 0x08,
     651,  599,   11,   11, 0x08,
     706,  698,   11,   11, 0x08,
     736,  698,   11,   11, 0x08,
     762,  698,   11,   11, 0x08,
     788,  698,   11,   11, 0x08,
     818,  698,   11,   11, 0x08,
     844,  698,   11,   11, 0x08,
     876,  870,   11,   11, 0x08,
     911,  870,   11,   11, 0x08,
     944,  870,   11,   11, 0x08,
     979,  870,   11,   11, 0x08,
    1014,  870,   11,   11, 0x08,
    1047,  870,   11,   11, 0x08,
    1082,  870,   11,   11, 0x08,
    1122,   11,   11,   11, 0x08,
    1156,   11,   11,   11, 0x08,
    1184,   11,   11,   11, 0x08,
    1208,   11,   11,   11, 0x08,
    1238,   11,   11,   11, 0x08,
    1258,   11,   11,   11, 0x08,
    1285,  870,   11,   11, 0x08,
    1318,   11,   11,   11, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_MainWindow[] = {
    "MainWindow\0\0step\0on_received_timestep(int)\0"
    "source\0on_received_datasource(QString)\0"
    "on_transferFunctionA_changed()\0"
    "on_transferFunctionB_changed()\0"
    "on_correct_sidebar_positions()\0"
    "triggerDataSidebar()\0triggerTransferfuncSidebar()\0"
    "triggerSettingsSidebar()\0triggerInfoBar()\0"
    "triggerClipBar()\0triggerUi()\0"
    "hideAllSidebars()\0triggerSimulationGallery()\0"
    "changeBackgroundcolor()\0uri\0"
    "connectToURI(QString)\0"
    "received_datasource(QString)\0"
    "received_timestep(int)\0id,size,data\0"
    "sendMessage(uint,uint,const void*)\0"
    "transferFunctionA_changed()\0"
    "transferFunctionB_changed()\0playPauseSim()\0"
    "clippingChanged()\0refresh_glWidget()\0"
    "arg1\0on_cmbDatasourceA_currentIndexChanged(QString)\0"
    "on_cmbDatasourceB_currentIndexChanged(QString)\0"
    "checked\0on_rdoRedGreenA_toggled(bool)\0"
    "on_rdoTempA_toggled(bool)\0"
    "on_rdo2HueA_toggled(bool)\0"
    "on_rdoRedGreenB_toggled(bool)\0"
    "on_rdoTempB_toggled(bool)\0"
    "on_rdo2HueB_toggled(bool)\0value\0"
    "on_sldOffsetXTFA_valueChanged(int)\0"
    "on_sldSlopeTFA_valueChanged(int)\0"
    "on_sldOffsetYTFA_valueChanged(int)\0"
    "on_sldOffsetXTFB_valueChanged(int)\0"
    "on_sldSlopeTFB_valueChanged(int)\0"
    "on_sldOffsetYTFB_valueChanged(int)\0"
    "on_sldSourceWeighting_valueChanged(int)\0"
    "on_sldOffsetXTFA_sliderReleased()\0"
    "correct_sidebar_positions()\0"
    "on_pushButton_clicked()\0"
    "on_rdoAlphaBlending_clicked()\0"
    "on_rdoMIP_clicked()\0on_rdoIsoSurface_clicked()\0"
    "on_sldIsoValue_valueChanged(int)\0"
    "on_btnWritePng_clicked()\0"
};

void MainWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        MainWindow *_t = static_cast<MainWindow *>(_o);
        switch (_id) {
        case 0: _t->on_received_timestep((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: _t->on_received_datasource((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 2: _t->on_transferFunctionA_changed(); break;
        case 3: _t->on_transferFunctionB_changed(); break;
        case 4: _t->on_correct_sidebar_positions(); break;
        case 5: _t->triggerDataSidebar(); break;
        case 6: _t->triggerTransferfuncSidebar(); break;
        case 7: _t->triggerSettingsSidebar(); break;
        case 8: _t->triggerInfoBar(); break;
        case 9: _t->triggerClipBar(); break;
        case 10: _t->triggerUi(); break;
        case 11: _t->hideAllSidebars(); break;
        case 12: _t->triggerSimulationGallery(); break;
        case 13: _t->changeBackgroundcolor(); break;
        case 14: _t->connectToURI((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 15: _t->received_datasource((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 16: _t->received_timestep((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 17: _t->sendMessage((*reinterpret_cast< uint(*)>(_a[1])),(*reinterpret_cast< uint(*)>(_a[2])),(*reinterpret_cast< const void*(*)>(_a[3]))); break;
        case 18: _t->transferFunctionA_changed(); break;
        case 19: _t->transferFunctionB_changed(); break;
        case 20: _t->playPauseSim(); break;
        case 21: _t->clippingChanged(); break;
        case 22: _t->refresh_glWidget(); break;
        case 23: _t->on_cmbDatasourceA_currentIndexChanged((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 24: _t->on_cmbDatasourceB_currentIndexChanged((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 25: _t->on_rdoRedGreenA_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 26: _t->on_rdoTempA_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 27: _t->on_rdo2HueA_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 28: _t->on_rdoRedGreenB_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 29: _t->on_rdoTempB_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 30: _t->on_rdo2HueB_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 31: _t->on_sldOffsetXTFA_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 32: _t->on_sldSlopeTFA_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 33: _t->on_sldOffsetYTFA_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 34: _t->on_sldOffsetXTFB_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 35: _t->on_sldSlopeTFB_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 36: _t->on_sldOffsetYTFB_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 37: _t->on_sldSourceWeighting_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 38: _t->on_sldOffsetXTFA_sliderReleased(); break;
        case 39: _t->correct_sidebar_positions(); break;
        case 40: _t->on_pushButton_clicked(); break;
        case 41: _t->on_rdoAlphaBlending_clicked(); break;
        case 42: _t->on_rdoMIP_clicked(); break;
        case 43: _t->on_rdoIsoSurface_clicked(); break;
        case 44: _t->on_sldIsoValue_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 45: _t->on_btnWritePng_clicked(); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData MainWindow::staticMetaObjectExtraData = {
    0,  qt_static_metacall
};

const QMetaObject MainWindow::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_MainWindow,
      qt_meta_data_MainWindow, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &MainWindow::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *MainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *MainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_MainWindow))
        return static_cast<void*>(const_cast< MainWindow*>(this));
    if (!strcmp(_clname, "rivlib::image_stream_connection::listener"))
        return static_cast< rivlib::image_stream_connection::listener*>(const_cast< MainWindow*>(this));
    if (!strcmp(_clname, "rivlib::control_connection::listener"))
        return static_cast< rivlib::control_connection::listener*>(const_cast< MainWindow*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int MainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 46)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 46;
    }
    return _id;
}

// SIGNAL 0
void MainWindow::on_received_timestep(int _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void MainWindow::on_received_datasource(QString _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void MainWindow::on_transferFunctionA_changed()
{
    QMetaObject::activate(this, &staticMetaObject, 2, 0);
}

// SIGNAL 3
void MainWindow::on_transferFunctionB_changed()
{
    QMetaObject::activate(this, &staticMetaObject, 3, 0);
}

// SIGNAL 4
void MainWindow::on_correct_sidebar_positions()
{
    QMetaObject::activate(this, &staticMetaObject, 4, 0);
}
QT_END_MOC_NAMESPACE
