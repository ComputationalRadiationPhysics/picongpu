/****************************************************************************
** Meta object code from reading C++ file 'simulationgallery.h'
**
** Created: Wed Jan 8 16:15:08 2014
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "simulationgallery.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'simulationgallery.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_SimulationGallery[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       4,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: signature, parameters, type, tag, flags
      19,   18,   18,   18, 0x05,
      36,   32,   18,   18, 0x05,

 // slots: signature, parameters, type, tag, flags
      65,   56,   18,   18, 0x0a,
     105,   18,   18,   18, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_SimulationGallery[] = {
    "SimulationGallery\0\0on_hide_me()\0uri\0"
    "connect_to(QString)\0name,uri\0"
    "add_vis_name_uri_thumb(QString,QString)\0"
    "on_refreshing_list()\0"
};

void SimulationGallery::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        SimulationGallery *_t = static_cast<SimulationGallery *>(_o);
        switch (_id) {
        case 0: _t->on_hide_me(); break;
        case 1: _t->connect_to((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 2: _t->add_vis_name_uri_thumb((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2]))); break;
        case 3: _t->on_refreshing_list(); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData SimulationGallery::staticMetaObjectExtraData = {
    0,  qt_static_metacall
};

const QMetaObject SimulationGallery::staticMetaObject = {
    { &PictureFlow::staticMetaObject, qt_meta_stringdata_SimulationGallery,
      qt_meta_data_SimulationGallery, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &SimulationGallery::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *SimulationGallery::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *SimulationGallery::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_SimulationGallery))
        return static_cast<void*>(const_cast< SimulationGallery*>(this));
    return PictureFlow::qt_metacast(_clname);
}

int SimulationGallery::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = PictureFlow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 4)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 4;
    }
    return _id;
}

// SIGNAL 0
void SimulationGallery::on_hide_me()
{
    QMetaObject::activate(this, &staticMetaObject, 0, 0);
}

// SIGNAL 1
void SimulationGallery::connect_to(QString _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}
QT_END_MOC_NAMESPACE
