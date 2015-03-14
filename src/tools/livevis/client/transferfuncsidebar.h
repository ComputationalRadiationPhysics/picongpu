#ifndef TRANSFERFUNCSIDEBAR_H
#define TRANSFERFUNCSIDEBAR_H

#include <QWidget>

namespace Ui {
class TransferfuncSidebar;
}

class TransferfuncSidebar : public QWidget
{
    Q_OBJECT

public:
    explicit TransferfuncSidebar(QWidget *parent = 0);
    ~TransferfuncSidebar();

private:
    Ui::TransferfuncSidebar *ui;
};

#endif // TRANSFERFUNCSIDEBAR_H
