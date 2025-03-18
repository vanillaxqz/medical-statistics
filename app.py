import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QMessageBox,
    QDockWidget, QWidget, QVBoxLayout, QPushButton, QScrollArea,
    QToolBar, QInputDialog, QLabel, QFileDialog,
    QToolButton, QMenu
)
from PyQt6.QtCore import Qt
import plots
import pandas as pd


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Medical Statistics")
        self.setMinimumSize(1000, 600)

        # toolbar
        toolbar = QToolBar("Toolbar")
        toolbar.setMovable(False)
        action_import = toolbar.addAction("Import Data")
        action_import.triggered.connect(self.import_data)

        toolbar.addAction("New Spreadsheet")

        action_save = toolbar.addAction("Export File")
        action_save.triggered.connect(self.export_file)

        insertButton = QToolButton()
        insertButton.setText("Insert")
        insertButton.clicked.connect(self.insert_column)
        insertMenu = QMenu(insertButton)
        actionInsertRow = insertMenu.addAction("Insert Row")
        actionInsertColumn = insertMenu.addAction("Insert Column")
        insertButton.setMenu(insertMenu)
        insertButton.setPopupMode(
            QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        actionInsertRow.triggered.connect(self.insert_row)
        actionInsertColumn.triggered.connect(self.insert_column)
        toolbar.addWidget(insertButton)

        deleteButton = QToolButton()
        deleteButton.setText("Delete")
        deleteButton.clicked.connect(self.delete_column)
        deleteMenu = QMenu(deleteButton)
        actionDeleteRow = deleteMenu.addAction("Delete Row")
        actionDeleteColumn = deleteMenu.addAction("Delete Column")
        deleteButton.setMenu(deleteMenu)
        deleteButton.setPopupMode(
            QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        actionDeleteRow.triggered.connect(self.delete_row)
        actionDeleteColumn.triggered.connect(self.delete_column)
        toolbar.addWidget(deleteButton)

        self.addToolBar(toolbar)

        # spreadsheet
        self.table = QTableWidget(15, 8)
        headers = [f"var{i}" for i in range(1, 9)]
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().sectionDoubleClicked.connect(self.rename_column)
        self.setCentralWidget(self.table)

        # sidebar
        dock = QDockWidget(self)
        dock.setFixedWidth(100)
        sidebar_title = QLabel("Plots")
        sidebar_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dock.setTitleBarWidget(sidebar_title)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        dock_layout = QVBoxLayout()

        histogram_button = QPushButton("Histogram")
        histogram_button.clicked.connect(self.plot_histogram)
        dock_layout.addWidget(histogram_button)

        boxplot_button = QPushButton("Box Plot")
        boxplot_button.clicked.connect(self.plot_box_plot)
        dock_layout.addWidget(boxplot_button)

        dock_layout.addWidget(QPushButton("Scatter Plot"))
        dock_layout.addWidget(QPushButton("Pie Chart"))

        dock_layout.addStretch()

        dock_container = QWidget()
        dock_container.setLayout(dock_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidget(dock_container)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        dock.setWidget(scroll_area)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def insert_row(self):
        current_row_count = self.table.rowCount()
        self.table.insertRow(current_row_count)

    def insert_column(self):
        current_column_count = self.table.columnCount()
        self.table.insertColumn(current_column_count)
        self.table.setHorizontalHeaderItem(
            current_column_count, QTableWidgetItem(f"var{current_column_count + 1}"))

    def delete_row(self):
        current_row_count = self.table.rowCount()
        if current_row_count > 0:
            self.table.removeRow(current_row_count - 1)

    def delete_column(self):
        current_column_count = self.table.columnCount()
        if current_column_count > 0:
            self.table.removeColumn(current_column_count - 1)

    def import_data(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "", "Excel Files (*.xlsx);;CSV Files (*.csv)"
        )
        if not file_name:
            return

        try:
            if file_name.endswith('.csv'):
                df = pd.read_csv(file_name, header=0)
            elif file_name.endswith('.xlsx'):
                df = pd.read_excel(file_name, header=0)
            else:
                QMessageBox.warning(self, "Unsupported File",
                                    "Please select a .csv or .xlsx file.")
                return
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}")
            return

        self.table.clear()
        self.table.setColumnCount(df.shape[1])
        self.table.setRowCount(df.shape[0])
        self.table.setHorizontalHeaderLabels(list(df.columns))
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                item = QTableWidgetItem(str(df.iat[i, j]))
                self.table.setItem(i, j, item)

    def export_file(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Data File", "", "Excel Files (*.xlsx);;CSV Files (*.csv)"
        )
        if not file_name:
            return

        rows = self.table.rowCount()
        cols = self.table.columnCount()

        headers = []
        for j in range(cols):
            header_item = self.table.horizontalHeaderItem(j)
            if header_item is not None:
                headers.append(header_item.text())
            else:
                headers.append(f"var{j+1}")

        data = []
        for i in range(rows):
            row_data = []
            for j in range(cols):
                item = self.table.item(i, j)
                row_data.append(item.text() if item is not None else "")
            data.append(row_data)

        df = pd.DataFrame(data, columns=headers)

        try:
            if file_name.endswith('.csv'):
                df.to_csv(file_name, index=False)
            elif file_name.endswith('.xlsx'):
                df.to_excel(file_name, index=False)
            else:
                df.to_csv(file_name + ".csv", index=False)
            QMessageBox.information(
                self, "Export Successful", "File exported successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed",
                                 f"Failed to export file:\n{e}")

    def rename_column(self, index):
        item = self.table.horizontalHeaderItem(index)
        text = item.text() if item else f"var{index + 1}"

        new_text, ok = QInputDialog.getText(
            self, "Rename Column", "Enter new column name:", text=text)
        if ok and new_text:
            self.table.horizontalHeaderItem(index).setText(new_text)

    def plot_histogram(self):
        headers = [self.table.horizontalHeaderItem(
            i).text() for i in range(self.table.columnCount())]
        chosen, ok = QInputDialog.getItem(
            self, "Select Variable", "Choose variable for histogram:", headers, 0, False)
        if not ok:
            return

        col_index = headers.index(chosen)
        numeric_data = []
        text_data = []

        for row in range(self.table.rowCount()):
            item = self.table.item(row, col_index)
            if item is not None:
                cell_text = item.text().strip()
                try:
                    value = float(cell_text)
                    numeric_data.append(value)
                except ValueError:
                    if cell_text:
                        text_data.append(cell_text)

        if not numeric_data and not text_data:
            QMessageBox.warning(
                self, "No Data", f"No valid data found in column '{chosen}'.")
            return

        plots.plot_histogram(chosen, numeric_data, text_data)

    def plot_box_plot(self):
        headers = [self.table.horizontalHeaderItem(
            i).text() for i in range(self.table.columnCount())]

        numeric_col, ok = QInputDialog.getItem(
            self, "Select Numeric Column", "Choose numeric variable for box plot:", headers, 0, False)
        if not ok:
            return

        group_option, ok = QInputDialog.getItem(
            self, "Group Data?", "Do you want to group data by another column?", ["No", "Yes"], 0, False)
        if not ok:
            return

        if group_option == "Yes":
            grouping_col, ok = QInputDialog.getItem(
                self, "Select Grouping Column", "Choose grouping variable for box plot:", headers, 0, False)
            if not ok:
                return

            numeric_index = headers.index(numeric_col)
            grouping_index = headers.index(grouping_col)
            grouped_data = {}

            for row in range(self.table.rowCount()):
                num_item = self.table.item(row, numeric_index)
                grp_item = self.table.item(row, grouping_index)
                if num_item is not None and grp_item is not None:
                    num_text = num_item.text().strip()
                    grp_text = grp_item.text().strip()
                    try:
                        value = float(num_text)
                        if grp_text:
                            grouped_data.setdefault(grp_text, []).append(value)
                    except ValueError:
                        pass

            if not grouped_data:
                QMessageBox.warning(
                    self, "No Data", f"No valid grouped data found for box plot.")
                return

            plots.plot_grouped_box_plot(
                numeric_col, grouping_col, grouped_data)
        else:
            numeric_index = headers.index(numeric_col)
            numeric_data = []
            for row in range(self.table.rowCount()):
                item = self.table.item(row, numeric_index)
                if item is not None:
                    cell_text = item.text().strip()
                    try:
                        value = float(cell_text)
                        numeric_data.append(value)
                    except ValueError:
                        pass
            if not numeric_data:
                QMessageBox.warning(
                    self, "No Data", f"No valid numeric data found in column '{numeric_col}'.")
                return

            plots.plot_box_plot(numeric_col, numeric_data)


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())
