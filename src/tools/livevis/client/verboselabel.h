#ifndef VERBOSELABEL_H
#define VERBOSELABEL_H

#include <QLabel>

class VerboseLabel : public QLabel
{
    Q_OBJECT

public:
    explicit VerboseLabel(QWidget * parent = 0);

    virtual void setPixmap(const QPixmap& pm)
    {
        emit on_set_pixmap();
        QLabel::setPixmap(pm);
    }

signals:

    void on_set_pixmap();

public slots:

};

#endif // VERBOSELABEL_H
