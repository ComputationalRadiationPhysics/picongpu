#include "datasourcesidebar.h"
#include "ui_datasourcesidebar.h"

#include <iostream>
#include <unistd.h>
#include <QMoveEvent>

DataSourceSidebar::DataSourceSidebar(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::DataSourceSidebar)
{
    ui->setupUi(this);
}

DataSourceSidebar::~DataSourceSidebar()
{
    delete ui;
}

void DataSourceSidebar::moveEvent(QMoveEvent * me)
{
    this->parentWidget()->update();
    this->update();

    std::cout << "Moved Event raised!" << std::endl;
}
