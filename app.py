import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QMessageBox,
    QDockWidget, QWidget, QVBoxLayout, QPushButton, QScrollArea,
    QToolBar, QInputDialog, QLabel, QFileDialog, QMessageBox,
    QToolButton, QMenu, QInputDialog, QDialog, QVBoxLayout,
    QWidget, QCheckBox, QDialogButtonBox, QHBoxLayout,
    QFormLayout, QComboBox, QLineEdit, QTextEdit
)
from PyQt6.QtGui import QKeySequence, QShortcut, QTextCursor, QFont
from PyQt6.QtCore import Qt
import plots
import markdown
import pandas as pd
import numpy as np
import matplotlib.pyplot as _plt
import re
from google import genai
from google.genai import types


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Medical Statistics")
        self.setMinimumSize(1000, 600)

        # assistant info
        self.client = genai.Client(
            api_key="AIzaSyB7iAlzXI84MnhXeSZCKaUeOzZtXoMzRnU")

        full_sys = (
            "You are the in-app assistant for the Medical Statistics desktop application. "
            "This app provides these built-in tools: Histogram, Box Plot, Scatter Plot (1D, 2D, Overlaid), Pie Chart, "
            "Bar Chart (single and grouped), Scatter Matrix, Line Graph, Kaplan–Meier Survival Plot, K-Means (elbow and clustering), "
            "Linear Regression, Exponential Regression, One-Way ANOVA, and Binary Logistic Regression.\n\n"
            "Behavior:\n"
            "1. When asked about a built-in function, give a concise description and instructions for invoking it.\n"
            "2. When asked for medical use-cases of a supported plot, offer one or two scenarios with a concrete example "
            "(e.g., “Use a scatter plot to compare fasting glucose vs. insulin in diabetic patients.”).\n"
            "3. If the user explicitly requests to run or generate an analysis not available via the toolbar "
            "(keywords like “run,” “generate,” or “implement”), immediately return ONLY a self-contained Python snippet that:\n"
            "   • Assumes a pandas DataFrame named df with numeric columns.\n"
            "   • Dynamically reads the specified columns, for example:\n"
            "       ```python\n"
            "       cols = ['col1', 'col2']\n"
            "       data = {c: df[c].astype(float).tolist() for c in cols}\n"
            "       ```\n"
            "   • Imports needed libraries (pandas, numpy, scipy.optimize.curve_fit, matplotlib.pyplot).\n"
            "   • Fits the requested model and plots it.\n"
            "   • Inserts exactly these lines before plt.show():\n"
            "       ```python\n"
            "       manager = plt.get_current_fig_manager()\n"
            "       manager.window.setWindowTitle(\"Assistant plot\")\n"
            "       plots.save_on_close(plt.gcf(), \"assisted_plot\", subfolder=\"gen_plots\")\n"
            "       ```\n"
            "   • Calls plt.show() immediately after.\n"
            "4. Never respond with “not implemented” or ask for clarification when the user requests an unsupported analysis—always provide the code snippet as specified.\n"
            "5. If the user asks what alternatives or other features they can use to achieve a task not directly supported, suggest existing tools or workflows in the app that could accomplish a similar goal, with brief instructions on using them."
        )

        try:
            self.chat_session = self.client.chats.create(
                model="gemini-2.0-flash",
                history=[types.ModelContent(parts=[types.Part(text=full_sys)])]
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Assistant Startup Failed",
                f"Could not connect to the assistant:\n{e}"
            )
            self.chat_input.setEnabled(False)
            chat_send_btn.setEnabled(False)
            return

        # toolbar - facem actions (butoane) si le adaugam in toolbar, conectam fiecare buton la o functie
        toolbar = QToolBar("Toolbar")
        toolbar.setMovable(False)

        action_import = toolbar.addAction("Import Data")
        action_import.triggered.connect(self.import_data)

        clear_action = toolbar.addAction("Reset Spreadsheet")
        clear_action.triggered.connect(self.clear_spreadsheet)

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

        transposeButton = toolbar.addAction("Transpose")
        transposeButton.triggered.connect(self.transpose_table)

        set_vheaders = toolbar.addAction("Set Row Headers")
        set_vheaders.triggered.connect(self.set_column_as_vheaders)

        set_headers = toolbar.addAction("Set Column Headers")
        set_headers.triggered.connect(self.set_row_as_hheaders)

        chat_toggle = toolbar.addAction("Assistant")
        chat_toggle.triggered.connect(
            lambda: self.chat_dock.setVisible(not self.chat_dock.isVisible()))

        self.addToolBar(toolbar)

        # chat widget
        self.chat_dock = QDockWidget("Assistant", self)
        self.chat_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.chat_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.chat_dock.setVisible(False)

        chat_container = QWidget()
        chat_layout = QVBoxLayout(chat_container)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setPlaceholderText(
            "Chat history will appear here...")
        self.chat_input = QLineEdit()
        chat_send_btn = QPushButton("Send")
        self.chat_input.setFixedHeight(self.chat_input.sizeHint().height() * 2)
        chat_send_btn.setFixedSize(60, self.chat_input.sizeHint().height() * 2)

        chat_send_btn.clicked.connect(self.handle_chat)
        self.chat_input.returnPressed.connect(self.handle_chat)

        chat_layout.addWidget(self.chat_display)

        input_row = QHBoxLayout()
        input_row.addWidget(self.chat_input)
        input_row.addWidget(chat_send_btn)
        chat_layout.addLayout(input_row)
        # chat_layout.addWidget(self.chat_input)
        # chat_layout.addWidget(chat_send_btn)

        self.chat_dock.setWidget(chat_container)
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.chat_dock)

        # spreadsheet
        self.table = QTableWidget(15, 8)
        headers = [f"var{i}" for i in range(1, 9)]
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().sectionDoubleClicked.connect(self.rename_column)
        self.table.verticalHeader().sectionDoubleClicked.connect(self.rename_row)
        self.setCentralWidget(self.table)

        # shortcuts:
        # ctrl c, ctrl v, ctrl x au regular behaviour
        # ctrl b pentru a sterge randuri sau coloane selectate
        self.delete_sc = QShortcut(QKeySequence("Ctrl+B"), self.table)
        self.delete_sc.activated.connect(self.delete_selected)

        self.copy_sc = QShortcut(QKeySequence(
            QKeySequence.StandardKey.Copy),  self.table)
        self.copy_sc.activated.connect(self.copy_selection)
        self.paste_sc = QShortcut(QKeySequence(
            QKeySequence.StandardKey.Paste), self.table)
        self.paste_sc.activated.connect(self.paste_selection)
        self.cut_sc = QShortcut(
            QKeySequence(QKeySequence.StandardKey.Cut),
            self.table
        )
        self.cut_sc.activated.connect(self.cut_selection)

        self._clipboard_data = []

        # sidebar widget
        # sunt create butoane, conectate la o functie si adaugate in widget din stanga
        dock = QDockWidget(self)
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

        scatter_button = QPushButton("Scatter Plot")
        scatter_button.clicked.connect(self.plot_scatter)
        dock_layout.addWidget(scatter_button)

        pie_button = QPushButton("Pie Chart")
        pie_button.clicked.connect(self.plot_pie_chart)
        dock_layout.addWidget(pie_button)

        bar_button = QPushButton("Bar Chart")
        bar_button.clicked.connect(self.plot_bar_chart)
        dock_layout.addWidget(bar_button)

        scatter_mat_button = QPushButton("Scatter Matrix")
        scatter_mat_button.clicked.connect(self.plot_scatter_matrix)
        dock_layout.addWidget(scatter_mat_button)

        survival_button = QPushButton("Survival Plot")
        survival_button.clicked.connect(self.plot_survival)
        dock_layout.addWidget(survival_button)

        linegraph_button = QPushButton("Line Graph")
        linegraph_button.clicked.connect(self.plot_line_graph)
        dock_layout.addWidget(linegraph_button)

        anova_btn = QPushButton("ANOVA")
        anova_btn.clicked.connect(self.run_anova)
        dock_layout.addWidget(anova_btn)

        linregression_button = QPushButton("Linear Regression")
        linregression_button.clicked.connect(self.run_linear_regression)
        dock_layout.addWidget(linregression_button)

        logistic_reg_btn = QPushButton("Binary Log. Regression")
        logistic_reg_btn.clicked.connect(self.run_logistic)
        dock_layout.addWidget(logistic_reg_btn)

        exp_reg_btn = QPushButton("Exponential Regression")
        exp_reg_btn.clicked.connect(self.run_exp_regression)
        dock_layout.addWidget(exp_reg_btn)

        kmeans_button = QPushButton("K-Means")
        kmeans_button.clicked.connect(self.run_kmeans)
        dock_layout.addWidget(kmeans_button)
        # generate_data = QPushButton("Generate Data")
        # generate_data.clicked.connect(self.generate_missing_data)
        # dock_layout.addWidget(generate_data)

        dock_layout.addStretch()

        dock_container = QWidget()
        dock_container.setLayout(dock_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidget(dock_container)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        dock.setWidget(scroll_area)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def clear_spreadsheet(self):
        self.table.clear()
        self.table.setRowCount(15)
        self.table.setColumnCount(8)
        headers = [f"var{i}" for i in range(1, 9)]
        self.table.setHorizontalHeaderLabels(headers)

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

    def prompt_missing_column(self, col: str, frac: float):
        # prompt pentru cazul in care o coloana are >= 30% cells empty
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Column '{col}' missing {int(frac*100)}% data")
        layout = QVBoxLayout(dlg)
        label = QLabel(
            f"Column '{col}' has {frac:.0%} missing values. What would you like to do?")
        layout.addWidget(label)

        btns = QDialogButtonBox(dlg)
        keep_btn = btns.addButton(
            "Keep", QDialogButtonBox.ButtonRole.AcceptRole)
        drop_btn = btns.addButton(
            "Drop", QDialogButtonBox.ButtonRole.DestructiveRole)
        fill_btn = btns.addButton(
            "Fill...", QDialogButtonBox.ButtonRole.ActionRole)
        layout.addWidget(btns)

        keep_btn.clicked.connect(lambda: dlg.done(1))
        drop_btn.clicked.connect(lambda: dlg.done(2))
        fill_btn.clicked.connect(lambda: dlg.done(3))

        result = dlg.exec()
        if result == 1:
            return 'keep', None
        if result == 2:
            return 'drop', None
        if result == 3:
            val, ok = QInputDialog.getText(
                self, "Fill Missing", f"Enter value to fill for '{col}':")
            if ok:
                return 'fill', val
        return 'keep', None

    def load_excel_file(self, file_path: str) -> pd.DataFrame:
        raw = pd.read_excel(file_path, header=None)
        # incercam sa gasim primul rand care are cel putin 2 valori care nu sunt
        # not a number pt header
        header_idx = next((i for i, row in raw.iterrows()
                          if row.notna().sum() > 1), 0)
        df = pd.read_excel(file_path, header=header_idx)
        df = df.dropna(axis=1, how='all')
        # daca sunt alte celule in acest header care sunt nan sau se repeta le eliminam
        cols_norm = df.columns.astype(str).str.strip().tolist()
        col_set = set(cols_norm)

        # functie care identifica headere care se repeta (subtabele)
        def is_repeat_header(row):
            vals = row.astype(str).str.strip().tolist()
            matches = sum(val in col_set for val in vals)
            return matches >= (len(cols_norm) / 2)

        # inversam masca ca sa le eliminam
        df = df[~df.apply(is_repeat_header, axis=1)]
        df = df.dropna(axis=0, how='all').reset_index(drop=True)
        return df

    def import_data(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "", "Excel & CSV Files (*.xlsx *.csv)"
        )
        if not file_name:
            return

        try:
            if file_name.lower().endswith('.csv'):
                df = pd.read_csv(file_name, header=0)
            else:
                df = self.load_excel_file(file_name)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}")
            return

        self.display_dataframe(df)

        # daca coloana are >= 30% nan ii dam prompt userului.
        for col in list(df.columns):
            frac = df[col].isna().mean()
            if frac >= 0.3:
                action, value = self.prompt_missing_column(col, frac)
                if action == 'drop':
                    df.drop(columns=[col], inplace=True)
                elif action == 'fill':
                    df[col].fillna(value, inplace=True)
                self.display_dataframe(df)

    def display_dataframe(self, df: pd.DataFrame):
        self.table.clear()
        self.table.setColumnCount(df.shape[1])
        self.table.setRowCount(df.shape[0])
        self.table.setHorizontalHeaderLabels(list(df.columns))
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                val = df.iat[i, j]
                text = '' if (pd.isna(val)) else str(val)
                item = QTableWidgetItem(text)
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

    def set_column_as_vheaders(self):
        col_labels = [self.table.horizontalHeaderItem(i).text()
                      for i in range(self.table.columnCount())]

        dlg = QDialog(self)
        dlg.setWindowTitle("Select Column as Vertical Header")
        layout = QVBoxLayout(dlg)

        combo = QComboBox(dlg)
        combo.addItems(col_labels)
        layout.addWidget(QLabel("Choose a column:"))
        layout.addWidget(combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            selected_col = combo.currentText()
            col_index = col_labels.index(selected_col)

            for row in range(self.table.rowCount() - 1, -1, -1):
                item = self.table.takeItem(row, col_index)
                if item:
                    self.table.setVerticalHeaderItem(
                        row, QTableWidgetItem(item.text()))
                else:
                    self.table.removeRow(row)

            self.table.removeColumn(col_index)

    def set_row_as_hheaders(self):
        row_labels = [
            (self.table.verticalHeaderItem(i).text()
             if self.table.verticalHeaderItem(i) else str(i+1))
            for i in range(self.table.rowCount())
        ]

        dlg = QDialog(self)
        dlg.setWindowTitle("Select Row as Column Headers")
        layout = QVBoxLayout(dlg)

        combo = QComboBox(dlg)
        combo.addItems(row_labels)
        layout.addWidget(QLabel("Choose a row:"))
        layout.addWidget(combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        selected_row = combo.currentText()
        row_index = row_labels.index(selected_row)

        for col in range(self.table.columnCount() - 1, -1, -1):
            item = self.table.takeItem(row_index, col)
            if item and item.text().strip():
                self.table.setHorizontalHeaderItem(
                    col, QTableWidgetItem(item.text()))
            else:
                self.table.removeColumn(col)

        self.table.removeRow(row_index)

    # shortcut functions #
    def copy_selection(self):
        ranges = self.table.selectedRanges()
        if not ranges:
            return
        r = ranges[0]
        rows = r.rowCount()
        cols = r.columnCount()
        data = []
        for i in range(rows):
            row = []
            for j in range(cols):
                item = self.table.item(r.topRow()+i, r.leftColumn()+j)
                row.append(item.text() if item else "")
            data.append(row)
        self._clipboard_data = data
        text = "\n".join("\t".join(r) for r in data)
        QApplication.clipboard().setText(text)

    def paste_selection(self):
        if not self._clipboard_data:
            return
        start_row = self.table.currentRow()
        start_col = self.table.currentColumn()
        if start_row < 0 or start_col < 0:
            return
        for i, row in enumerate(self._clipboard_data):
            if start_row + i >= self.table.rowCount():
                self.table.insertRow(self.table.rowCount())
            for j, val in enumerate(row):
                if start_col + j >= self.table.columnCount():
                    self.table.insertColumn(self.table.columnCount())
                self.table.setItem(start_row+i, start_col +
                                   j, QTableWidgetItem(val))

    def delete_selected(self):
        sel = self.table.selectionModel()
        rows = sel.selectedRows()
        if rows:
            for idx in sorted((r.row() for r in rows), reverse=True):
                self.table.removeRow(idx)
            return
        cols = sel.selectedColumns()
        if cols:
            for idx in sorted((c.column() for c in cols), reverse=True):
                self.table.removeColumn(idx)

    def cut_selection(self):
        self.copy_selection()

        ranges = self.table.selectedRanges()
        if not ranges:
            return
        r = ranges[0]
        for i in range(r.rowCount()):
            for j in range(r.columnCount()):
                row = r.topRow() + i
                col = r.leftColumn() + j
                item = self.table.item(row, col)
                if item:
                    item.setText("")
                else:
                    self.table.setItem(row, col, QTableWidgetItem(""))

    def transpose_table(self):
        rows = self.table.rowCount()
        cols = self.table.columnCount()

        col_labels = [
            self.table.horizontalHeaderItem(j).text()
            if self.table.horizontalHeaderItem(j) else f"var{j+1}"
            for j in range(cols)
        ]
        row_labels = [
            self.table.verticalHeaderItem(i).text()
            if self.table.verticalHeaderItem(i) else str(i+1)
            for i in range(rows)
        ]

        data = {
            col_labels[j]: [
                (self.table.item(i, j).text() if self.table.item(i, j) else "")
                for i in range(rows)
            ]
            for j in range(cols)
        }
        df = pd.DataFrame(data, index=row_labels)

        df_t = df.transpose()

        new_col_labels = row_labels
        new_row_labels = col_labels

        self.table.clear()
        self.table.setRowCount(len(new_row_labels))
        self.table.setColumnCount(len(new_col_labels))

        self.table.setHorizontalHeaderLabels(new_col_labels)
        self.table.setVerticalHeaderLabels(new_row_labels)

        for i, rlabel in enumerate(new_row_labels):
            for j, clabel in enumerate(new_col_labels):
                val = df_t.iat[i, j]
                self.table.setItem(i, j, QTableWidgetItem(str(val)))

    def rename_column(self, index):
        item = self.table.horizontalHeaderItem(index)
        text = item.text() if item else f"var{index + 1}"

        new_text, ok = QInputDialog.getText(
            self, "Rename Column", "Enter new column name:", text=text)
        if ok and new_text:
            self.table.horizontalHeaderItem(index).setText(new_text)

    def rename_row(self, index):
        item = self.table.verticalHeaderItem(index)
        text = item.text() if item else f"Row {index + 1}"

        new_text, ok = QInputDialog.getText(
            self, "Rename Row", "Enter new row name:", text=text)
        if ok and new_text:
            self.table.setVerticalHeaderItem(index, QTableWidgetItem(new_text))

    # functie care creeaza un pop up unde putem selecta coloane si randuri pentru ploturi
    def choose_columns_and_rows(self, column_labels, row_labels, dialog_title="Select Columns and Rows"):
        dlg = QDialog(self)
        dlg.setWindowTitle(dialog_title)
        layout = QVBoxLayout(dlg)

        scroll_layout = QHBoxLayout()

        def make_scrollable_checkbox_list(labels, title):
            group = QVBoxLayout()
            group.setAlignment(Qt.AlignmentFlag.AlignTop)
            group.addWidget(QLabel(f"<b>{title}</b>"))
            checks = []
            for label in labels:
                cb = QCheckBox(label)
                group.addWidget(cb)
                checks.append(cb)

            container = QWidget()
            container.setLayout(group)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(container)
            scroll.setMinimumHeight(200)
            scroll.setMaximumHeight(300)
            return scroll, checks

        col_scroll, col_checks = make_scrollable_checkbox_list(
            column_labels, "Columns")
        row_scroll, row_checks = make_scrollable_checkbox_list(
            row_labels, "Rows")

        scroll_layout.addWidget(col_scroll)
        scroll_layout.addWidget(row_scroll)
        layout.addLayout(scroll_layout)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        dlg.setMinimumSize(600, 400)

        selected_cols, selected_rows = [], []

        if dlg.exec() == QDialog.DialogCode.Accepted:
            selected_cols = [cb.text() for cb in col_checks if cb.isChecked()]
            selected_rows = [cb.text() for cb in row_checks if cb.isChecked()]

        return selected_cols, selected_rows

    def plot_histogram(self):
        col_labels = [self.table.horizontalHeaderItem(i).text()
                      for i in range(self.table.columnCount())]
        row_labels = [self.table.verticalHeaderItem(i).text()
                      if self.table.verticalHeaderItem(i) else str(i+1)
                      for i in range(self.table.rowCount())]

        cols, rows = self.choose_columns_and_rows(
            col_labels, row_labels, "Select Columns and Rows for Histogram")
        if not cols and not rows:
            QMessageBox.warning(
                self, "Selection Required",
                "Please select at least one column or one row for histogram."
            )
            return

        row_map = {label: idx for idx, label in enumerate(row_labels)}
        selected_row_indices = (
            [row_map[r] for r in rows] if rows
            else list(range(self.table.rowCount()))
        )

        numeric_data_dict = {}
        text_data_dict = {}

        for col in cols:
            idx = col_labels.index(col)
            nums, texts = [], []
            for r in selected_row_indices:
                item = self.table.item(r, idx)
                if item:
                    txt = item.text().strip()
                    try:
                        nums.append(float(txt))
                    except ValueError:
                        if txt:
                            texts.append(txt)
            if nums and texts:
                QMessageBox.warning(
                    self, "Mixed Data Types",
                    f"Column '{col}' contains both numeric and text values."
                )
                return
            if nums:
                numeric_data_dict[col] = nums
            elif texts:
                text_data_dict[col] = texts

        for row_label in rows:
            r = row_map[row_label]
            nums, texts = [], []
            for c in range(self.table.columnCount()):
                item = self.table.item(r, c)
                if item:
                    txt = item.text().strip()
                    try:
                        nums.append(float(txt))
                    except ValueError:
                        if txt:
                            texts.append(txt)
            if nums and texts:
                QMessageBox.warning(
                    self, "Mixed Data Types",
                    f"Row '{row_label}' contains both numeric and text values."
                )
                return
            if nums:
                numeric_data_dict[row_label] = nums
            elif texts:
                text_data_dict[row_label] = texts

        if not numeric_data_dict:
            QMessageBox.warning(
                self, "No Numeric Data",
                "Please select at least one column or row with numeric values."
            )
            return

        keys = list(numeric_data_dict.keys())
        if len(keys) == 1:
            k = keys[0]
            plots.plot_histogram(k, numeric_data_dict[k], [])
        else:
            plots.plot_histogram(keys, numeric_data_dict, {})

    def plot_box_plot(self):
        col_labels = [self.table.horizontalHeaderItem(i).text()
                      for i in range(self.table.columnCount())]
        row_labels = [self.table.verticalHeaderItem(i).text() if self.table.verticalHeaderItem(i) else str(i+1)
                      for i in range(self.table.rowCount())]

        cols, rows = self.choose_columns_and_rows(
            col_labels, row_labels,
            "Select one or more columns or rows for Box Plot"
        )
        if not cols and not rows:
            QMessageBox.warning(self, "Selection Required",
                                "Please select at least one column or one row for box plot.")
            return

        row_map = {label: idx for idx, label in enumerate(row_labels)}
        numeric_data = {}

        for col in cols:
            idx = col_labels.index(col)
            vals = []
            for r in range(self.table.rowCount()):
                item = self.table.item(r, idx)
                if item and self._is_float(item.text().strip()):
                    vals.append(float(item.text().strip()))
            if vals:
                numeric_data[col] = vals

        for row_label in rows:
            ridx = row_map[row_label]
            vals = []
            for c in range(self.table.columnCount()):
                item = self.table.item(ridx, c)
                if item and self._is_float(item.text().strip()):
                    vals.append(float(item.text().strip()))
            if vals:
                numeric_data[row_label] = vals

        if not numeric_data:
            QMessageBox.warning(self, "No Data",
                                "None of the selected columns or rows contained numeric data.")
            return

        if len(numeric_data) > 1:
            plots.plot_box_plots(
                list(numeric_data.keys()), numeric_data
            )
            return

        label, data = next(iter(numeric_data.items()))

        if label in cols:
            grouping_cols, _ = self.choose_columns_and_rows(
                col_labels, [], "Select zero or more columns to group by"
            )
            grouping_cols = [c for c in grouping_cols if c != label]
            if not grouping_cols:
                plots.plot_box_plot(label, data)
                return

            grouped = {}
            for r in range(self.table.rowCount()):
                try:
                    key = " | ".join(
                        self.table.item(r, col_labels.index(gc)).text().strip()
                        for gc in grouping_cols
                    )
                    val = float(self.table.item(
                        r, col_labels.index(label)).text().strip())
                    grouped.setdefault(key, []).append(val)
                except:
                    continue
            if not grouped:
                QMessageBox.warning(
                    self, "No Data", "No valid grouped data found for box plot.")
                return
            plots.plot_grouped_box_plot(
                label, ", ".join(grouping_cols), grouped)

        else:
            _, grouping_rows = self.choose_columns_and_rows(
                [], row_labels, "Select zero or more rows to group by"
            )
            grouping_rows = [r for r in grouping_rows if r != label]
            if not grouping_rows:
                plots.plot_box_plot(label, data)
                return

            primary_idx = row_map[label]
            grouped = {}
            for c in range(self.table.columnCount()):
                try:
                    key = " | ".join(
                        self.table.item(row_map[grp], c).text().strip()
                        for grp in grouping_rows
                    )
                    val = float(self.table.item(primary_idx, c).text().strip())
                    grouped.setdefault(key, []).append(val)
                except:
                    continue
            if not grouped:
                QMessageBox.warning(
                    self, "No Data", "No valid grouped data found for box plot.")
                return
            plots.plot_grouped_box_plot(
                label, ", ".join(grouping_rows), grouped)

    def plot_scatter(self):
        col_labels = [self.table.horizontalHeaderItem(i).text()
                      for i in range(self.table.columnCount())]
        row_labels = [self.table.verticalHeaderItem(i).text() if self.table.verticalHeaderItem(i) else str(i+1)
                      for i in range(self.table.rowCount())]

        kind, ok = QInputDialog.getItem(
            self, "Scatter Type",
            "Choose scatter dimension:", ["1D", "2D"], 0, False
        )
        if not ok:
            return

        if kind == "1D":
            cols, rows = self.choose_columns_and_rows(
                col_labels, row_labels,
                "Select one or more columns or rows for 1D scatter"
            )
            if not cols and not rows:
                QMessageBox.warning(self, "Selection Required",
                                    "Please select at least one column or row.")
                return

            data_dict = {}
            for col in cols:
                idx = col_labels.index(col)
                vals = [float(self.table.item(r, idx).text())
                        for r in range(self.table.rowCount())
                        if self.table.item(r, idx) and self._is_float(self.table.item(r, idx).text())]
                if vals:
                    data_dict[col] = vals

            for row in rows:
                ridx = row_labels.index(row)
                vals = [float(self.table.item(ridx, c).text())
                        for c in range(self.table.columnCount())
                        if self.table.item(ridx, c) and self._is_float(self.table.item(ridx, c).text())]
                if vals:
                    data_dict[row] = vals

            if not data_dict:
                QMessageBox.warning(self, "No Data",
                                    "None of the selected columns or rows contained numeric data.")
                return

            plots.plot_scatter_1d(data_dict)
            return

        # 2d scatter case
        count, ok = QInputDialog.getInt(
            self, "2D Scatter Pairs",
            "How many X–Y pairs would you like?", 1, 1, 10, 1)
        if not ok:
            return

        pairs = []
        for i in range(count):
            x_cols, x_rows = self.choose_columns_and_rows(
                col_labels, row_labels,
                f"Pair {i+1}: select exactly one column or row for X axis"
            )
            if len(x_cols) + len(x_rows) != 1:
                QMessageBox.warning(self, "Select X Variable",
                                    "Please select exactly one column or row.")
                return
            if x_cols:
                x_label = x_cols[0]
                x_idx = col_labels.index(x_label)
                x_data = [float(self.table.item(r, x_idx).text())
                          for r in range(self.table.rowCount())
                          if self.table.item(r, x_idx) and self._is_float(self.table.item(r, x_idx).text())]
            else:
                x_label = x_rows[0]
                ridx = row_labels.index(x_label)
                x_data = [float(self.table.item(ridx, c).text())
                          for c in range(self.table.columnCount())
                          if self.table.item(ridx, c) and self._is_float(self.table.item(ridx, c).text())]

            y_cols, y_rows = self.choose_columns_and_rows(
                col_labels, row_labels,
                f"Pair {i+1}: select exactly one column or row for Y axis"
            )
            if len(y_cols) + len(y_rows) != 1:
                QMessageBox.warning(self, "Select Y Variable",
                                    "Please select exactly one column or row.")
                return
            if y_cols:
                y_label = y_cols[0]
                y_idx = col_labels.index(y_label)
                y_data = [float(self.table.item(r, y_idx).text())
                          for r in range(self.table.rowCount())
                          if self.table.item(r, y_idx) and self._is_float(self.table.item(r, y_idx).text())]
            else:
                y_label = y_rows[0]
                ridx = row_labels.index(y_label)
                y_data = [float(self.table.item(ridx, c).text())
                          for c in range(self.table.columnCount())
                          if self.table.item(ridx, c) and self._is_float(self.table.item(ridx, c).text())]

            if not x_data or not y_data:
                QMessageBox.warning(self, "No Data",
                                    f"Pair {i+1}: could not extract numeric data.")
                return
            n = min(len(x_data), len(y_data))
            pairs.append((x_label, y_label, x_data[:n], y_data[:n]))

        if count == 1:
            x_label, y_label, x_vals, y_vals = pairs[0]
            plots.plot_scatter(x_label, y_label, x_vals, y_vals)
        else:
            plots.plot_scatter_overlaid(pairs)

    def plot_pie_chart(self):
        col_labels = [
            self.table.horizontalHeaderItem(i).text()
            for i in range(self.table.columnCount())]
        row_labels = [
            self.table.verticalHeaderItem(
                i).text() if self.table.verticalHeaderItem(i) else str(i+1)
            for i in range(self.table.rowCount())]

        cols, rows = self.choose_columns_and_rows(
            col_labels, row_labels,
            "Select columns and/or rows for pie charts")
        if not cols and not rows:
            QMessageBox.warning(
                self, "Selection Required",
                "Please select at least one column or one row.")
            return

        pies = []
        for col in cols:
            idx = col_labels.index(col)
            counts = {}
            for r in range(self.table.rowCount()):
                item = self.table.item(r, idx)
                if item:
                    txt = item.text().strip()
                    if txt:
                        counts[txt] = counts.get(txt, 0) + 1
            if counts:
                pies.append((col, counts))

        for row in rows:
            ridx = row_labels.index(row)
            counts = {}
            for c in range(self.table.columnCount()):
                item = self.table.item(ridx, c)
                if item:
                    txt = item.text().strip()
                    if txt:
                        counts[txt] = counts.get(txt, 0) + 1
            if counts:
                pies.append((row, counts))

        if not pies:
            QMessageBox.warning(
                self, "No Data",
                "None of the selected columns or rows contained any data.")
            return

        if len(pies) == 1:
            label, counts = pies[0]
            plots.plot_pie_chart(label, counts)
        else:
            plots.plot_pie_charts(pies)

    def plot_bar_chart(self):
        col_labels = [self.table.horizontalHeaderItem(i).text()
                      for i in range(self.table.columnCount())]
        row_labels = [(self.table.verticalHeaderItem(i).text()
                       if self.table.verticalHeaderItem(i) else str(i+1))
                      for i in range(self.table.rowCount())]

        cols, rows = self.choose_columns_and_rows(
            col_labels, row_labels,
            "Select one or more numeric columns or rows for the bar chart"
        )
        total = len(cols) + len(rows)
        if total < 1:
            QMessageBox.warning(self, "Need Variables",
                                "Please select at least one column or one row.")
            return

        if cols and rows:
            labels, means = [], []
            for col in cols:
                idx = col_labels.index(col)
                vals = [float(self.table.item(r, idx).text())
                        for r in range(self.table.rowCount())
                        if self.table.item(r, idx) and self._is_float(self.table.item(r, idx).text())]
                if not vals:
                    QMessageBox.warning(self, "Non-Numeric Data",
                                        f"Column '{col}' has no numeric data.")
                    return
                labels.append(col)
                means.append(sum(vals)/len(vals))
            for row_label in rows:
                ridx = row_labels.index(row_label)
                vals = [float(self.table.item(ridx, c).text())
                        for c in range(self.table.columnCount())
                        if self.table.item(ridx, c) and self._is_float(self.table.item(ridx, c).text())]
                if not vals:
                    QMessageBox.warning(self, "Non-Numeric Data",
                                        f"Row '{row_label}' has no numeric data.")
                    return
                labels.append(row_label)
                means.append(sum(vals)/len(vals))

            plots.plot_bar_chart(labels, means)
            return

        if cols:
            numeric_cols = cols
            grouping_cols, _ = self.choose_columns_and_rows(
                col_labels, [], "Select zero or more columns to group by"
            )
            if not grouping_cols:
                labels, means = [], []
                for col in numeric_cols:
                    idx = col_labels.index(col)
                    vals = [float(self.table.item(r, idx).text())
                            for r in range(self.table.rowCount())
                            if self.table.item(r, idx) and self._is_float(self.table.item(r, idx).text())]
                    if not vals:
                        QMessageBox.warning(self, "Non-Numeric Data",
                                            f"Column '{col}' has no numeric data.")
                        return
                    labels.append(col)
                    means.append(sum(vals)/len(vals))
                plots.plot_bar_chart(labels, means)
                return

            grouped = {}
            group_keys = []
            for r in range(self.table.rowCount()):
                parts = []
                for g in grouping_cols:
                    cidx = col_labels.index(g)
                    itm = self.table.item(r, cidx)
                    parts.append(itm.text().strip()
                                 if itm and itm.text().strip() else "")
                if not all(parts):
                    continue
                key = " | ".join(parts)
                if key not in grouped:
                    grouped[key] = [[] for _ in numeric_cols]
                    group_keys.append(key)
                for i, col in enumerate(numeric_cols):
                    cidx = col_labels.index(col)
                    itm = self.table.item(r, cidx)
                    if itm and itm.text().strip() and self._is_float(itm.text()):
                        grouped[key][i].append(float(itm.text()))

            labels = numeric_cols
            means_dict = {}
            for key in group_keys:
                lists = grouped[key]
                for lst in lists:
                    if not lst:
                        QMessageBox.warning(self, "Incomplete Data",
                                            f"No numeric data for group '{key}'.")
                        return
                means_dict[key] = [sum(lst)/len(lst) for lst in lists]

            plots.plot_grouped_bar_chart(
                labels, group_keys, means_dict,
                grouping_vars=grouping_cols
            )
            return

        if rows:
            numeric_rows = rows
            _, grouping_rows = self.choose_columns_and_rows(
                [], row_labels, "Select zero or more rows to group by"
            )
            if not grouping_rows:
                labels, means = [], []
                for row_label in numeric_rows:
                    ridx = row_labels.index(row_label)
                    vals = [float(self.table.item(ridx, c).text())
                            for c in range(self.table.columnCount())
                            if self.table.item(ridx, c) and self._is_float(self.table.item(ridx, c).text())]
                    if not vals:
                        QMessageBox.warning(self, "Non-Numeric Data",
                                            f"Row '{row_label}' has no numeric data.")
                        return
                    labels.append(row_label)
                    means.append(sum(vals)/len(vals))
                plots.plot_bar_chart(labels, means)
                return

            grouped = {}
            group_keys = []
            for c in range(self.table.columnCount()):
                parts = []
                for g in grouping_rows:
                    ridx = row_labels.index(g)
                    itm = self.table.item(ridx, c)
                    parts.append(itm.text().strip()
                                 if itm and itm.text().strip() else "")
                if not all(parts):
                    continue
                key = " | ".join(parts)
                if key not in grouped:
                    grouped[key] = [[] for _ in numeric_rows]
                    group_keys.append(key)
                for i, row_label in enumerate(numeric_rows):
                    ridx = row_labels.index(row_label)
                    itm = self.table.item(ridx, c)
                    if itm and itm.text().strip() and self._is_float(itm.text()):
                        grouped[key][i].append(float(itm.text()))

            labels = numeric_rows
            means_dict = {}
            for key in group_keys:
                lists = grouped[key]
                for lst in lists:
                    if not lst:
                        QMessageBox.warning(self, "Incomplete Data",
                                            f"No numeric data for group '{key}'.")
                        return
                means_dict[key] = [sum(lst)/len(lst) for lst in lists]

            plots.plot_grouped_bar_chart(
                labels, group_keys, means_dict,
                grouping_vars=grouping_rows
            )
            return

    def plot_scatter_matrix(self):
        col_labels = [self.table.horizontalHeaderItem(i).text()
                      for i in range(self.table.columnCount())]
        row_labels = [
            (self.table.verticalHeaderItem(i).text()
             if self.table.verticalHeaderItem(i) else str(i+1))
            for i in range(self.table.rowCount())
        ]

        cols, rows = self.choose_columns_and_rows(
            col_labels, row_labels,
            "Select one or more numeric columns and/or rows for Scatter Matrix"
        )
        if not cols and not rows:
            QMessageBox.warning(
                self, "Selection Required",
                "Please select at least one column or one row."
            )
            return

        numeric_data = {}

        for col in cols:
            idx = col_labels.index(col)
            vals = []
            for r in range(self.table.rowCount()):
                txt = self.table.item(r, idx).text().strip()
                if self._is_float(txt):
                    vals.append(float(txt))
            if vals:
                numeric_data[col] = vals

        row_map = {lbl: i for i, lbl in enumerate(row_labels)}
        for row_lbl in rows:
            ridx = row_map[row_lbl]
            vals = []
            for c in range(self.table.columnCount()):
                txt = self.table.item(ridx, c).text().strip()
                if self._is_float(txt):
                    vals.append(float(txt))
            if vals:
                numeric_data[row_lbl] = vals

        if len(numeric_data) < 2:
            QMessageBox.warning(
                self, "Insufficient Variables",
                "Need at least two numeric series (e.g. 2 columns, 2 rows, or 1 of each) "
                "for a scatter-matrix."
            )
            return

        min_len = min(len(v) for v in numeric_data.values())
        for lbl in numeric_data:
            numeric_data[lbl] = numeric_data[lbl][:min_len]

        plots.plot_scatter_matrix(list(numeric_data.keys()), numeric_data)

    def _is_float(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def plot_survival(self):
        cols = [self.table.horizontalHeaderItem(i).text()
                for i in range(self.table.columnCount())]
        dlg = QDialog(self)
        dlg.setWindowTitle("Survival Plot Options")
        layout = QFormLayout(dlg)

        group_cb = QComboBox(dlg)
        group_cb.addItem("<None>")
        group_cb.addItems(cols)
        layout.addRow("Group (optional):", group_cb)

        time_cb = QComboBox(dlg)
        time_cb.addItems(cols)
        layout.addRow("Time column:", time_cb)

        event_cb = QComboBox(dlg)
        event_cb.addItems(cols)
        layout.addRow("Event column (0=censor,1=event):", event_cb)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel,
            parent=dlg)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addRow(buttons)
        dlg.setLayout(layout)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        group_col = group_cb.currentText()
        if group_col == "<None>":
            group_col = None
        time_col = time_cb.currentText()
        event_col = event_cb.currentText()

        data = []
        for r in range(self.table.rowCount()):
            row = [
                self.table.item(r, c).text() if self.table.item(r, c) else ""
                for c in range(self.table.columnCount())
            ]
            data.append(row)
        df = pd.DataFrame(data, columns=cols)

        try:
            df[time_col] = pd.to_numeric(df[time_col], errors="raise")

            raw = df[event_col]
            if raw.dtype == object or pd.api.types.is_string_dtype(raw):
                df[event_col] = (
                    raw.astype(str)
                       .str.strip()
                       .str.lower()
                       .map({"true": True, "false": False, "1": True, "0": False})
                       .fillna(False)
                )
            else:
                df[event_col] = pd.to_numeric(raw, errors="raise").astype(bool)

        except Exception as e:
            QMessageBox.critical(
                self, "Conversion Error",
                f"Error converting columns: {e}")
            return

        plots.plot_kaplan_meier(df, time_col, event_col, group_col)

    def run_kmeans(self):
        cols = [self.table.horizontalHeaderItem(c).text()
                for c in range(self.table.columnCount())]
        dlg = QDialog(self)
        dlg.setWindowTitle("Select Axes")
        form = QFormLayout(dlg)

        cb_x = QComboBox(dlg)
        cb_x.addItems(cols)
        cb_y = QComboBox(dlg)
        cb_y.addItems(cols)
        form.addRow("X axis:", cb_x)
        form.addRow("Y axis:", cb_y)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel,
            parent=dlg
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        dlg.setLayout(form)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        x_col, y_col = cb_x.currentText(), cb_y.currentText()
        if x_col == y_col:
            QMessageBox.warning(self, "Invalid selection",
                                "Please pick two diff columns.")
            return

        max_k, ok = QInputDialog.getInt(
            self, "Max Clusters",
            "Compute inertia up to K =",
            value=10, min=1, max=100, step=1
        )
        if not ok:
            return

        data = []
        for r in range(self.table.rowCount()):
            row = [
                self.table.item(r, c).text() if self.table.item(r, c) else ""
                for c in range(self.table.columnCount())
            ]
            data.append(row)
        df = pd.DataFrame(data, columns=cols)

        plots.plot_kmeans_elbow(df, x_col, y_col, max_k)

        while _plt.get_fignums():
            _plt.pause(0.1)

        k, ok = QInputDialog.getInt(
            self, "Choose K",
            "How many clusters?",
            value=3, min=1, max=max_k, step=1
        )
        if not ok:
            return

        plots.plot_kmeans_scatter(df, x_col, y_col, k)

    def run_linear_regression(self):
        cols = [
            self.table.horizontalHeaderItem(c).text()
            for c in range(self.table.columnCount())
        ]
        dlg = QDialog(self)
        dlg.setWindowTitle("Linear Regression: Select Axes")
        form = QFormLayout(dlg)

        cb_x = QComboBox(dlg)
        cb_x.addItems(cols)
        cb_y = QComboBox(dlg)
        cb_y.addItems(cols)
        form.addRow("X:", cb_x)
        form.addRow("Y:", cb_y)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel,
            parent=dlg
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        dlg.setLayout(form)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        x_col = cb_x.currentText()
        y_col = cb_y.currentText()
        if x_col == y_col:
            QMessageBox.warning(self, "Invalid selection",
                                "Please pick two different columns.")
            return

        data = []
        for r in range(self.table.rowCount()):
            row = [
                self.table.item(r, c).text() if self.table.item(r, c) else ""
                for c in range(self.table.columnCount())
            ]
            data.append(row)
        df = pd.DataFrame(data, columns=cols)

        plots.plot_linear_regression(df, x_col, y_col)

    def plot_line_graph(self):
        row_labels = [
            (self.table.verticalHeaderItem(i).text()
             if self.table.verticalHeaderItem(i) else str(i+1))
            for i in range(self.table.rowCount())
        ]

        dlg = QDialog(self)
        dlg.setWindowTitle("Select rows for Line graph")
        dlg.resize(300, 300)
        main_layout = QVBoxLayout(dlg)

        scroll = QScrollArea(dlg)
        scroll.setWidgetResizable(True)
        container = QWidget()
        cb_layout = QVBoxLayout(container)
        cb_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        checkboxes = []
        for lbl in row_labels:
            cb = QCheckBox(lbl, container)
            cb_layout.addWidget(cb)
            checkboxes.append(cb)
        container.setLayout(cb_layout)
        scroll.setWidget(container)
        main_layout.addWidget(scroll)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel,
            parent=dlg
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        main_layout.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        selected = [cb.text() for cb in checkboxes if cb.isChecked()]
        if not selected:
            QMessageBox.warning(self, "Selection Required",
                                "Please select at least one row for the line graph.")
            return

        numeric_data = {}
        for lbl in selected:
            ridx = row_labels.index(lbl)
            vals = []
            for c in range(self.table.columnCount()):
                item = self.table.item(ridx, c)
                txt = item.text().strip() if item else ""
                if self._is_float(txt):
                    vals.append(float(txt))
            if vals:
                numeric_data[lbl] = vals

        if not numeric_data:
            QMessageBox.warning(self, "No Numeric Data",
                                "None of the selected rows contain numeric values.")
            return

        plots.plot_line_graph(list(numeric_data.keys()), numeric_data)

    def run_anova(self):
        row_labels = [
            str(self.table.model().headerData(r, Qt.Orientation.Vertical))
            for r in range(self.table.rowCount())
        ]

        mode, ok = QInputDialog.getItem(
            self, "ANOVA Mode", "Compute one-way ANOVA on",
            ["Rows", "Column grouped by row prefix"], 0, False)
        if not ok:
            return

        groups = {}

        if mode.startswith("Rows"):
            dlg = QDialog(self)
            dlg.setWindowTitle("Select Rows")
            dlg_layout = QVBoxLayout(dlg)

            scroll = QScrollArea(dlg)
            scroll.setWidgetResizable(True)
            scroll.setMaximumHeight(300)
            dlg_layout.addWidget(scroll)

            container = QWidget()
            scroll.setWidget(container)
            container_layout = QVBoxLayout(container)

            checkboxes = []
            for lbl in row_labels:
                cb = QCheckBox(lbl, container)
                container_layout.addWidget(cb)
                checkboxes.append(cb)

            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
                dlg
            )
            buttons.accepted.connect(dlg.accept)
            buttons.rejected.connect(dlg.reject)
            dlg_layout.addWidget(buttons)

            if dlg.exec() != QDialog.DialogCode.Accepted:
                return

            selected = [cb.text() for cb in checkboxes if cb.isChecked()]
            if len(selected) < 2:
                QMessageBox.warning(
                    self, "Need Multiple Rows",
                    "Please check at least two rows."
                )
                return

            for lbl in selected:
                r = row_labels.index(lbl)
                vals = []
                for c in range(self.table.columnCount()):
                    item = self.table.item(r, c)
                    txt = item.text().strip() if item else ""
                    if self._is_float(txt):
                        vals.append(float(txt))
                if vals:
                    groups[lbl] = vals

        else:
            col_labels = [str(self.table.model().headerData(c, Qt.Orientation.Horizontal))
                          for c in range(self.table.columnCount())]
            col_name, ok = QInputDialog.getItem(
                self, "Select Column",
                "Choose the measurement column:",
                col_labels, 0, False)
            if not ok:
                return
            col_idx = col_labels.index(col_name)

            for r in range(self.table.rowCount()):
                header = row_labels[r]
                item = self.table.item(r, col_idx)
                txt = item.text().strip() if item else ""
                if not self._is_float(txt):
                    continue
                m = re.match(r"^(\D+)", header)
                prefix = m.group(1) if m else header
                groups.setdefault(prefix, []).append(float(txt))

        groups = {k: v for k, v in groups.items() if v}
        if len(groups) < 2:
            QMessageBox.warning(
                self, "Not Enough Data",
                "Need at least two groups with numeric data.")
            return

        plots.perform_anova(groups)

    def run_logistic(self):
        cols = [self.table.horizontalHeaderItem(i).text()
                for i in range(self.table.columnCount())]

        dlg = QDialog(self)
        dlg.setWindowTitle("Logistic Regression: Select Columns")
        layout = QFormLayout(dlg)

        cb_x = QComboBox(dlg)
        cb_x.addItems(cols)
        cb_y = QComboBox(dlg)
        cb_y.addItems(cols)
        layout.addRow("Feature (X):", cb_x)
        layout.addRow("Binary Label (Y):", cb_y)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=dlg
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addRow(buttons)
        dlg.setLayout(layout)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        x_col = cb_x.currentText()
        y_col = cb_y.currentText()
        if x_col == y_col:
            QMessageBox.warning(self, "Invalid Selection",
                                "X and Y must be different columns.")
            return

        data = []
        for r in range(self.table.rowCount()):
            row_data = [
                self.table.item(r, c).text() if self.table.item(r, c) else ""
                for c in range(self.table.columnCount())
            ]
            data.append(row_data)
        df = pd.DataFrame(data, columns=cols)

        df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
        df_clean = df.dropna(subset=[x_col, y_col])
        df_clean = df_clean[df_clean[y_col].isin([0, 1])]
        if df_clean.empty:
            QMessageBox.warning(
                self, "Invalid Data",
                "No valid numeric feature values or binary labels (0/1) found.")
            return

        plots.plot_logistic_regression(df_clean, x_col, y_col)

    def run_exp_regression(self):
        cols = [self.table.horizontalHeaderItem(
            i).text() for i in range(self.table.columnCount())]

        dlg = QDialog(self)
        dlg.setWindowTitle("Exponential Regression: Select Axes")
        form = QFormLayout(dlg)

        cb_x = QComboBox(dlg)
        cb_x.addItems(cols)
        cb_y = QComboBox(dlg)
        cb_y.addItems(cols)
        form.addRow("X:", cb_x)
        form.addRow("Y:", cb_y)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel,
            parent=dlg
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        dlg.setLayout(form)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        x_col = cb_x.currentText()
        y_col = cb_y.currentText()
        if x_col == y_col:
            QMessageBox.warning(self, "Invalid selection",
                                "Please pick two different columns.")
            return

        data = []
        for r in range(self.table.rowCount()):
            row = [self.table.item(r, c).text() if self.table.item(
                r, c) else "" for c in range(self.table.columnCount())]
            data.append(row)
        df = pd.DataFrame(data, columns=cols)

        try:
            df[x_col] = pd.to_numeric(df[x_col], errors="raise")
            df[y_col] = pd.to_numeric(df[y_col], errors="raise")
        except Exception as e:
            QMessageBox.critical(self, "Conversion Error",
                                 f"Error converting columns: {e}")
            return

        plots.plot_exponential_reg(df, x_col, y_col)

    def handle_chat(self):
        text = self.chat_input.text().strip()
        if not text:
            return

        user_html = (
            f'<span style="color:#aaffaa; font-weight:bold;">You:</span> {text}<br>'
        )
        self.chat_display.moveCursor(QTextCursor.MoveOperation.End)
        self.chat_display.insertHtml(user_html)
        self.chat_display.moveCursor(QTextCursor.MoveOperation.End)
        self.chat_input.clear()

        try:
            resp = self.chat_session.send_message(text)
            reply = resp.text.strip()
        except Exception as e:
            err_msg = f"Network error: {e}"
            QMessageBox.warning(self, "Assistant Error", err_msg)
            self.chat_display.insertHtml(f"<b>AI:</b> <i>{err_msg}</i></div>")
            self.chat_display.moveCursor(QTextCursor.MoveOperation.End)
            return

        if reply.startswith("```"):
            code = re.sub(r"^```(?:python)?|```$", "",
                          reply, flags=re.I).strip()
            self._exec_snippet(code)
        else:
            md_html = markdown.markdown(reply, extensions=['extra', 'nl2br'])
            ai_html = ('<span style="color:#aaaaff; font-weight:bold;">AI:</span>'
                       f'{md_html}<br>')
            self.chat_display.moveCursor(QTextCursor.MoveOperation.End)
            self.chat_display.insertHtml(ai_html)
            self.chat_display.moveCursor(QTextCursor.MoveOperation.End)

    def _build_exec_namespace(self):
        import plots
        import pandas as pd

        headers = [self.table.horizontalHeaderItem(i).text()
                   for i in range(self.table.columnCount())]
        data = []
        for r in range(self.table.rowCount()):
            row = []
            for c in range(self.table.columnCount()):
                item = self.table.item(r, c)
                txt = item.text().strip() if item else ""
                row.append(txt)
            data.append(row)
        df = pd.DataFrame(data, columns=headers)

        df = df.apply(lambda col: pd.to_numeric(col, errors="ignore"))

        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        df[numeric_cols] = df[numeric_cols].replace(
            [plots.np.inf, -plots.np.inf], plots.np.nan)
        df = df.dropna(axis=0, subset=numeric_cols, how="all")

        return {
            "self": self,
            "df": df,
            "pd": pd,
            "plt": plots.plt,
            "curve_fit": plots.curve_fit,
            "LinearRegression": plots.LinearRegression,
            "LogisticRegression": plots.LogisticRegression,
            "f_oneway": plots.f_oneway,
            "KMeans": plots.KMeans,
            "plots": plots,
        }

    def _exec_snippet(self, code: str):
        ns = self._build_exec_namespace()
        try:
            exec(code, globals(), ns)
        except Exception as e:
            QMessageBox.critical(self, "Execution Error", str(e))


app = QApplication(sys.argv)
font = QFont("Segoe UI", 12)
app.setFont(font)
window = MainWindow()
window.show()
sys.exit(app.exec())
