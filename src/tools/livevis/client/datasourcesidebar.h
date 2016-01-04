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
