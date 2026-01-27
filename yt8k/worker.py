from PySide6.QtCore import QThread, Signal
from downloader import download_video

class DownloadWorker(QThread):
    progress = Signal(dict)
    error = Signal(str)
    finished = Signal()

    def __init__(self, url, path, res, codec):
        super().__init__()
        self.url = url
        self.path = path
        self.res = res
        self.codec = codec

    def run(self):
        try:
            download_video(
                self.url,
                self.path,
                self.res,
                self.codec,
                self.progress.emit
            )
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
