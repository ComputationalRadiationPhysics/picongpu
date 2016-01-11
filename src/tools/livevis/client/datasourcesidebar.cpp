/**
 * Copyright 2013-2016 Benjamin Schneider
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

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
