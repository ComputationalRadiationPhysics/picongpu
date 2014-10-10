#ifndef DATASOURCESIDEBAR_H
#define DATASOURCESIDEBAR_H

#include <QWidget>

namespace Ui {
class DataSourceSidebar;
}

class DataSourceSidebar : public QWidget
{
    Q_OBJECT

  public:
    explicit DataSourceSidebar(QWidget *parent = 0);
    ~DataSourceSidebar();

  protected:
    void moveEvent(QMoveEvent * me);

  private:
    Ui::DataSourceSidebar *ui;
};

#endif // DATASOURCESIDEBAR_H
