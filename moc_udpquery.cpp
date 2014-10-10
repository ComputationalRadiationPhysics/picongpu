/****************************************************************************
** Meta object code from reading C++ file 'udpquery.h'
**
** Created: Fri Oct 10 13:47:38 2014
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "udpquery.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'udpquery.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_UDPQuery[] = {

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
      19,   10,    9,    9, 0x05,
      59,    9,    9,    9, 0x05,

 // slots: signature, parameters, type, tag, flags
      77,    9,    9,    9, 0x0a,
      85,    9,    9,    9, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_UDPQuery[] = {
    "UDPQuery\0\0name,uri\0"
    "received_visualization(QString,QString)\0"
    "refreshing_list()\0start()\0refresh()\0"
};

void UDPQuery::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        UDPQuery *_t = static_cast<UDPQuery *>(_o);
        switch (_id) {
        case 0: _t->received_visualization((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2]))); break;
        case 1: _t->refreshing_list(); break;
        case 2: _t->start(); break;
        case 3: _t->refresh(); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData UDPQuery::staticMetaObjectExtraData = {
    0,  qt_static_metacall
};

const QMetaObject UDPQuery::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_UDPQuery,
      qt_meta_data_UDPQuery, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &UDPQuery::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *UDPQuery::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *UDPQuery::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_UDPQuery))
        return static_cast<void*>(const_cast< UDPQuery*>(this));
    return QObject::qt_metacast(_clname);
}

int UDPQuery::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
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
void UDPQuery::received_visualization(QString _t1, QString _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void UDPQuery::refreshing_list()
{
    QMetaObject::activate(this, &staticMetaObject, 1, 0);
}
QT_END_MOC_NAMESPACE
