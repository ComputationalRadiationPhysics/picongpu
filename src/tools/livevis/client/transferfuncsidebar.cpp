#include "transferfuncsidebar.h"
#include "ui_transferfuncsidebar.h"

TransferfuncSidebar::TransferfuncSidebar(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::TransferfuncSidebar)
{
    ui->setupUi(this);
}

TransferfuncSidebar::~TransferfuncSidebar()
{
    delete ui;
}
