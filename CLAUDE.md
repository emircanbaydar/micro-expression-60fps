# CLAUDE.md — 60 FPS Gerçek Zamanlı Mikro Mimik Analiz Sistemi

## Proje Özeti
60 FPS kamera akışı üzerinden mikro mimikleri (40–200 ms) tespit eden (Spotting) ve FACS Action Unit / duygu bazlı sınıflandıran (Recognition) gerçek zamanlı bilgisayarlı görü ve derin öğrenme pipeline'ı.

## Performans ve Donanım Kısıtları
- **Gerçek Zamanlı Bütçe:** Uçtan uca kare işleme süresi **16.6 ms'nin altında** (minimum 60 FPS) kalmalıdır.
- **Kamera Gecikme Koruması:** Kamera I/O işlemleri ana işlem döngüsünü (inference/flow) bloklamamalı; ayrı bir `threading.Thread` üzerinde en güncel kareyi tutacak şekilde çalışmalıdır.
- **Kamera Codec:** USB bant genişliği darboğazını (30 FPS kilidi) aşmak için `cv2.VideoCapture` mutlaka `MJPG` FOURCC formatı ve `CAP_PROP_FPS = 60` ile yapılandırılmalıdır.
- **Asenkron Sınıflandırma:** `SpotEvent` tetiklendiğinde STSTNet çıkarımı `concurrent.futures.ThreadPoolExecutor(max_workers=1)` ile arka planda çalışır. Ana video döngüsü tahmini beklemez; sonuç hazır olunca (`Future.done()`) HUD, banner ve log ana thread'de güncellenir. Önceki tahmin sürerken gelen olay için sınıflandırma atlanır (kuyruk şişmesini önlemek için).

## Teknoloji Yığını ve API Kuralları
- **Python:** 3.10+ (Tüm fonksiyonlarda Type Hints ve temiz docstring kullanılacak).
- **Yüz Landmark Tespiti:** Güncel `mediapipe.tasks.python.vision.FaceLandmarker` (`RunningMode.VIDEO`) API'si kullanılacak. Eski/deprecated `mp.solutions` kesinlikle kullanılmayacak.
- **Model Dosyası Yönetimi:** `models/face_landmarker.task` dosyası diskte yoksa uygulama başlangıcında resmi Google MediaPipe deposundan otomatik indirilmeli.
- **Hizalama:** 3 noktalı (göz merkezleri + burun ucu) Affine dönüşümüyle 256x256 yüz ROI'si üretilir. Landmark tespiti 480px genişliğe indirgenmiş kopyada yapılır.
- **Hareket Çıkarımı:** Hizalanmış 256x256 ROI'nin gri tonlamalı hâli **128x128** boyutuna indirgenir; Farnebäck Dense Optical Flow (`u`, `v`) ve uzamsal türevlerden **Optical Strain** bu **128x128** çözünürlükte hesaplanır. STSTNet tensörü `(3, 16, 128, 128)` boyutundadır. Göz kırpma gürültüsünü önlemek için göz kapağı bölgesi maskelenir.
- **Derin Öğrenme:** PyTorch tabanlı **STSTNet (Shallow Triple Stream 3D-CNN)** mimarisi kullanılacak; canlı çıkarım için ONNX Runtime (`FP16`/CPU-GPU provider) desteği sağlanacak.

## Veri Seti Düzeni
```text
data/
├── README.md
├── negative/<klip>/img001.jpg ...
├── positive/<klip>/img001.jpg ...
└── surprise/<klip>/img001.jpg ...
```
CASME II / SAMM kopyalama, manifest CSV'si ve özel `.mp4` ekleme ayrıntıları için `data/README.md` dosyasına bakın. Etiket çözümleme sırası `src/training/dataset.py` docstring'inde tanımlıdır.

## Komutlar
```text
# Video -> hizalanmış veri seti (256x256 kareler, img001.jpg ...)
python prepare_data.py --video klip.mp4 --label surprise
python prepare_data.py --input-dir ham_videolar/ --label positive
python prepare_data.py --video klip.mp4 --label negative --overwrite --stride 2

# STSTNet eğitimi (.pt + .onnx + .classes.txt üretir)
python train.py --dummy --epochs 5                 # boru hattı testi
python train.py --data-root data --epochs 50
python train.py --data-root data --manifest data/manifest.csv

# Canlı çalışma
python main.py                                     # models/ststnet.onnx otomatik
python main.py --video kayit.mp4 --realtime
python main.py --no-display --duration 10
python main.py --render-stride 3                   # görselleştirme her 3 karede bir
```

## Klasör Mimarisi (Modüler Yapı)
```text
micro-expression-60fps/
├── CLAUDE.md
├── requirements.txt
├── main.py                      # Ana çalışma döngüsü, FPS sayacı, asenkron STSTNet, OpenCV görselleştirme
├── prepare_data.py              # Video → hizalanmış veri seti CLI dönüştürücüsü
├── train.py                     # STSTNet eğitimi + ONNX dışa aktarımı
├── data/                        # Eğitim verisi (negative / positive / surprise)
├── models/                      # face_landmarker.task ve eğitilmiş .pt / .onnx ağırlıkları
└── src/
    ├── __init__.py
    ├── capture/
    │   └── camera_stream.py     # Threaded 60 FPS MJPG kamera okuyucu
    ├── preprocessing/
    │   ├── face_aligner.py      # MediaPipe FaceLandmarker + 3 noktalı Affine hizalama (256x256)
    │   └── optical_flow.py      # 128x128 Farnebäck Flow (u, v), Optical Strain ve göz kapağı maskeleme
    ├── models/
    │   └── ststnet.py           # Triple-stream (u, v, strain) 3D-CNN PyTorch mimarisi
    ├── training/
    │   └── dataset.py           # Klip keşfi, etiket çözümleme, (3, T, 128, 128) tensör üretimi
    └── pipeline/
        └── spotting_engine.py   # 16 karelik Ring Buffer (deque) + Apex eşik tetikleyici
```
