/****************************************************************************
** Meta object code from reading C++ file 'pictureflow.h'
**
** Created: Fri Oct 10 13:47:36 2014
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "pictureflow.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'pictureflow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_PictureFlow[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
      14,   14, // methods
       4,   84, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      19,   13,   12,   12, 0x05,

 // slots: signature, parameters, type, tag, flags
      49,   43,   12,   12, 0x0a,
      73,   66,   12,   12, 0x0a,
      91,   13,   12,   12, 0x0a,
     120,  108,   12,   12, 0x0a,
     154,  141,   12,   12, 0x0a,
     176,   13,   12,   12, 0x0a,
     196,   12,   12,   12, 0x0a,
     204,   12,   12,   12, 0x0a,
     219,   12,   12,   12, 0x0a,
     230,   13,   12,   12, 0x0a,
     245,   12,   12,   12, 0x0a,
     254,   12,   12,   12, 0x0a,
     270,   12,   12,   12, 0x08,

 // properties: name, type, flags
     295,  288, 0x43095103,
     317,  311, 0x15095103,
     331,  327, 0x02095001,
     342,  327, 0x02095103,

       0        // eod
};

static const char qt_meta_stringdata_PictureFlow[] = {
    "PictureFlow\0\0index\0centerIndexChanged(int)\0"
    "image\0addSlide(QImage)\0pixmap\0"
    "addSlide(QPixmap)\0removeSlide(int)\0"
    "index,image\0setSlide(int,QImage)\0"
    "index,pixmap\0setSlide(int,QPixmap)\0"
    "setCenterIndex(int)\0clear()\0showPrevious()\0"
    "showNext()\0showSlide(int)\0render()\0"
    "triggerRender()\0updateAnimation()\0"
    "QColor\0backgroundColor\0QSize\0slideSize\0"
    "int\0slideCount\0centerIndex\0"
};

void PictureFlow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        PictureFlow *_t = static_cast<PictureFlow *>(_o);
        switch (_id) {
        case 0: _t->centerIndexChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: _t->addSlide((*reinterpret_cast< const QImage(*)>(_a[1]))); break;
        case 2: _t->addSlide((*reinterpret_cast< const QPixmap(*)>(_a[1]))); break;
        case 3: _t->removeSlide((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 4: _t->setSlide((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< const QImage(*)>(_a[2]))); break;
        case 5: _t->setSlide((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< const QPixmap(*)>(_a[2]))); break;
        case 6: _t->setCenterIndex((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 7: _t->clear(); break;
        case 8: _t->showPrevious(); break;
        case 9: _t->showNext(); break;
        case 10: _t->showSlide((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 11: _t->render(); break;
        case 12: _t->triggerRender(); break;
        case 13: _t->updateAnimation(); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData PictureFlow::staticMetaObjectExtraData = {
    0,  qt_static_metacall
};

const QMetaObject PictureFlow::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_PictureFlow,
      qt_meta_data_PictureFlow, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &PictureFlow::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *PictureFlow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *PictureFlow::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_PictureFlow))
        return static_cast<void*>(const_cast< PictureFlow*>(this));
    return QWidget::qt_metacast(_clname);
}

int PictureFlow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 14)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 14;
    }
#ifndef QT_NO_PROPERTIES
      else if (_c == QMetaObject::ReadProperty) {
        void *_v = _a[0];
        switch (_id) {
        case 0: *reinterpret_cast< QColor*>(_v) = backgroundColor(); break;
        case 1: *reinterpret_cast< QSize*>(_v) = slideSize(); break;
        case 2: *reinterpret_cast< int*>(_v) = slideCount(); break;
        case 3: *reinterpret_cast< int*>(_v) = centerIndex(); break;
        }
        _id -= 4;
    } else if (_c == QMetaObject::WriteProperty) {
        void *_v = _a[0];
        switch (_id) {
        case 0: setBackgroundColor(*reinterpret_cast< QColor*>(_v)); break;
        case 1: setSlideSize(*reinterpret_cast< QSize*>(_v)); break;
        case 3: setCenterIndex(*reinterpret_cast< int*>(_v)); break;
        }
        _id -= 4;
    } else if (_c == QMetaObject::ResetProperty) {
        _id -= 4;
    } else if (_c == QMetaObject::QueryPropertyDesignable) {
        _id -= 4;
    } else if (_c == QMetaObject::QueryPropertyScriptable) {
        _id -= 4;
    } else if (_c == QMetaObject::QueryPropertyStored) {
        _id -= 4;
    } else if (_c == QMetaObject::QueryPropertyEditable) {
        _id -= 4;
    } else if (_c == QMetaObject::QueryPropertyUser) {
        _id -= 4;
    }
#endif // QT_NO_PROPERTIES
    return _id;
}

// SIGNAL 0
void PictureFlow::centerIndexChanged(int _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_END_MOC_NAMESPACE
