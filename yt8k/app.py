import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QLineEdit, QPushButton, QComboBox, QFileDialog,
    QProgressBar
)
from downloader import get_video_info
from worker import DownloadWorker


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YouTube 8K Downloader")
        self.resize(500, 400)

        self.layout = QVBoxLayout(self)

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("YouTube URL")

        self.fetch_btn = QPushButton("Fetch Video Info")

        self.res_box = QComboBox()
        self.codec_box = QComboBox()

        self.path_btn = QPushButton("Choose Save Folder")
        self.path_label = QLabel("")

        self.download_btn = QPushButton("Download")

        self.progress = QProgressBar()
        self.status = QLabel("")

        self.layout.addWidget(self.url_input)
        self.layout.addWidget(self.fetch_btn)
        self.layout.addWidget(self.res_box)
        self.layout.addWidget(self.codec_box)
        self.layout.addWidget(self.path_btn)
        self.layout.addWidget(self.path_label)
        self.layout.addWidget(self.download_btn)
        self.layout.addWidget(self.progress)
        self.layout.addWidget(self.status)

        self.fetch_btn.clicked.connect(self.fetch_info)
        self.path_btn.clicked.connect(self.pick_folder)
        self.download_btn.clicked.connect(self.start_download)

        self.video_data = {}
        self.save_path = ""

    def fetch_info(self):
        self.video_data = get_video_info(self.url_input.text())

        self.res_box.blockSignals(True)
        self.res_box.clear()
        self.res_box.addItems(self.video_data["resolutions"])
        self.res_box.blockSignals(False)

        self.res_box.currentTextChanged.connect(self.update_codecs)
        self.update_codecs()

    def update_codecs(self):
        self.codec_box.clear()
        res = self.res_box.currentText()

        codecs = self.video_data["codecs"].get(res)

        if not codecs:
            codecs = ["av1", "vp9", "avc1"]

        self.codec_box.addItems(codecs)

    def pick_folder(self):
        self.save_path = QFileDialog.getExistingDirectory(self)
        self.path_label.setText(self.save_path)

    def start_download(self):
        self.worker = DownloadWorker(
            self.url_input.text(),
            self.save_path,
            self.res_box.currentText(),
            self.codec_box.currentText()
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.error.connect(self.show_error)
        self.worker.finished.connect(lambda: self.status.setText("Done"))
        self.worker.start()

    def update_progress(self, data):
        if "percent" in data:
            self.progress.setValue(data["percent"])
        if data.get("finished"):
            self.status.setText("Download complete")

    def show_error(self, msg):
        self.status.setText(f"Error: {msg}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = App()
    win.show()
    sys.exit(app.exec())
