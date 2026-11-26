"""
PROJE-I  (tam surum)
  Kisim 1: Nesne tanima + kontur cizme
  Kisim 2: Nesneyi cikarma + bosluk doldurma + Farkli Kaydet
  Kisim 3: Disaridan nesne ekleme + renk/isik/golge uyumu
  Bonus  : Video / kamera uzerinde nesne tespiti

Gereksinimler: pip install ultralytics opencv-python numpy pillow

Mimari notu:
  Her nesne  {"ad": str, "maske": np.uint8 (H x W, 0/255)}  seklinde tutulur.
  YOLO da GrabCut da ayni formatta maske uretir.
  self.img     = ilk acilan resim (degismez)
  self.calisma = uzerinde calisilan, cikarma/ekleme sonrasi degisen resim
"""
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, colorchooser

import cv2
import numpy as np
from PIL import Image, ImageTk

MODEL_ADI = "yolov8m-seg.pt"   # onceden indirip proje klasorune koyun
CANVAS_MAX = (900, 650)        # ekranda gosterilecek en buyuk boyut

YONTEMLER = {
    "Akıllı doldurma (Inpaint)": "inpaint",
    "Çevre rengi (otomatik)": "cevre",
    "Seçtiğim renk": "renk",
}


# YARDIMCILAR
def resim_oku(yol):
    """png/jpg/bmp/tif hepsini 3 kanalli uint8 BGR'ye cevirir (Turkce yol da calisir)."""
    veri = np.fromfile(yol, dtype=np.uint8)
    img = cv2.imdecode(veri, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Resim okunamadı.")
    if img.dtype == np.uint16:                       # 16 bit tif
        img = (img / 256).astype(np.uint8)
    if img.ndim == 2:                                # gri
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:                          # alfa kanalli
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def tk_resim(bgr):
    """BGR resmi ekrana sigacak sekilde kucultup Tk resmine cevirir. (PhotoImage, olcek)"""
    h, w = bgr.shape[:2]
    olcek = min(CANVAS_MAX[0] / w, CANVAS_MAX[1] / h, 1.0)
    kucuk = cv2.resize(bgr, (max(1, int(w * olcek)), max(1, int(h * olcek))),
                       interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(kucuk, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb)), olcek


#  KISIM 1
def maske_temizle(maske, k=5):
    """Morfolojik kapama (delikleri kapat) + acma (kucuk lekeleri sil)."""
    cekirdek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    maske = cv2.morphologyEx(maske, cv2.MORPH_CLOSE, cekirdek)
    maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN, cekirdek)
    return maske


def kontur_ciz(img, maske, renk=(0, 255, 0), kalinlik=2):
    """Maskenin dis konturunu bulup resmin kopyasina cizer."""
    temiz = maske_temizle(maske)
    konturlar, _ = cv2.findContours(temiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cikti = img.copy()
    cv2.drawContours(cikti, konturlar, -1, renk, kalinlik, cv2.LINE_AA)
    return cikti


def grabcut_maske(img, kutu):
    """kutu = (x, y, w, h). Hizli olsun diye kucultup calistirir, maskeyi buyutur."""
    h, w = img.shape[:2]
    olcek = min(1.0, 800 / max(h, w))
    kucuk = cv2.resize(img, None, fx=olcek, fy=olcek, interpolation=cv2.INTER_AREA)
    x, y, bw, bh = [int(v * olcek) for v in kutu]
    gc = np.zeros(kucuk.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(kucuk, gc, (x, y, bw, bh), bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    kucuk_maske = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return cv2.resize(kucuk_maske, (w, h), interpolation=cv2.INTER_NEAREST)


# KISIM 2
def nesne_cikar(img, maske, yontem, genislet=9, renk=None):
    """
    Maskeyi genisletir (golge/kenar artigi kalmasin) ve boslugu doldurur.
      yontem: "inpaint" -> cevreden doku yayar (Telea)
              "cevre"   -> maskenin cevresindeki piksellerin medyan rengi
              "renk"    -> kullanicinin sectigi BGR renk
    """
    genislet = max(1, int(genislet))
    cekirdek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * genislet + 1, 2 * genislet + 1))
    genis = cv2.dilate(maske, cekirdek)
    h, w = img.shape[:2]

    if yontem == "inpaint":
        ys, xs = np.where(genis > 0)
        pad = 40
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, h)
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, w)
        parca = img[y0:y1, x0:x1]
        m = genis[y0:y1, x0:x1]
        ph, pw = parca.shape[:2]
        olcek = min(1.0, 700 / max(ph, pw))          # buyuk alanlarda hiz icin kucult
        if olcek < 1.0:
            kp = cv2.resize(parca, None, fx=olcek, fy=olcek, interpolation=cv2.INTER_AREA)
            km = cv2.resize(m, (kp.shape[1], kp.shape[0]), interpolation=cv2.INTER_NEAREST)
            dolu = cv2.inpaint(kp, km, 5, cv2.INPAINT_TELEA)
            dolu = cv2.resize(dolu, (pw, ph), interpolation=cv2.INTER_CUBIC)
        else:
            dolu = cv2.inpaint(parca, m, 5, cv2.INPAINT_TELEA)
        cikti = img.copy()
        alt = cikti[y0:y1, x0:x1]
        alt[m > 0] = dolu[m > 0]
        return cikti

    if yontem == "cevre":
        buyuk = cv2.dilate(genis, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * (genislet + 25) + 1, 2 * (genislet + 25) + 1)))
        halka = (buyuk > 0) & (genis == 0)
        renk = np.median(img[halka], axis=0) if halka.any() else (127, 127, 127)

    # duz renk dolgu, kenari hafif yumusatarak
    a = cv2.GaussianBlur(genis.astype(np.float32) / 255.0, (0, 0), 2)[..., None]
    cikti = img.astype(np.float32) * (1 - a) + np.array(renk, np.float32) * a
    return cikti.astype(np.uint8)


#  KISIM 3
def nesne_hazirla(yol):
    """
    Eklenecek resmi okur. Seffaf PNG ise alfa kanalini kullanir.
    Degilse kullanicidan kutu cizdirip GrabCut ile arka plani siler.
    Dondurur: (bgr, alfa)  -> nesnenin sinirlarina kirpilmis
    """
    veri = np.fromfile(yol, np.uint8)
    img = cv2.imdecode(veri, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Resim okunamadı.")
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)

    alfa = None
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        bgr, alfa = img[:, :, :3].copy(), img[:, :, 3].copy()
    else:
        bgr = img

    if alfa is None or alfa.min() > 250:             # seffaflik yok -> GrabCut
        h, w = bgr.shape[:2]
        olcek = min(1.0, 800 / max(h, w))
        kucuk = cv2.resize(bgr, None, fx=olcek, fy=olcek, interpolation=cv2.INTER_AREA)
        kutu = cv2.selectROI("Eklenecek nesneyi cevreleyen kutu cizin (ENTER = onay)", kucuk, False)
        cv2.destroyAllWindows()
        if kutu[2] < 5 or kutu[3] < 5:               # iptal -> tum resmi (kenar payi ile) al
            kutu = (int(0.02 * kucuk.shape[1]), int(0.02 * kucuk.shape[0]),
                    int(0.96 * kucuk.shape[1]), int(0.96 * kucuk.shape[0]))
        kutu = tuple(int(v / olcek) for v in kutu)
        alfa = grabcut_maske(bgr, kutu)

    ys, xs = np.where(alfa > 10)
    if len(ys) == 0:
        raise ValueError("Nesne ayrılamadı.")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return bgr[y0:y1, x0:x1].copy(), alfa[y0:y1, x0:x1].copy()


def renk_aktar(obj, alfa, hedef, guc):
    """
    Reinhard renk transferi (LAB uzayi): nesnenin ortalama/standart sapmasini
    hedef bolgeninkine yaklastirir. guc: 0 = hic, 1 = tam. L kanali isik uyumunu saglar.
    """
    if guc <= 0:
        return obj
    o = cv2.cvtColor(obj, cv2.COLOR_BGR2LAB).astype(np.float32)
    t = cv2.cvtColor(hedef, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3)
    m = alfa > 127
    if m.sum() < 10:
        return obj
    mu_o, sd_o = o[m].mean(0), o[m].std(0) + 1e-6
    mu_t, sd_t = t.mean(0), t.std(0) + 1e-6
    oran = np.clip(sd_t / sd_o, 0.6, 1.4)
    yeni = (o - mu_o) * oran + mu_t
    karisim = np.clip(o * (1 - guc) + yeni * guc, 0, 255).astype(np.uint8)
    return cv2.cvtColor(karisim, cv2.COLOR_LAB2BGR)


def birlestir(bg, obj, alfa, taban, oran, yansit=False, uyum=0.7,
              golge=True, golge_yon=0.5, poisson=False):
    """
    Nesneyi bg uzerine yerlestirir.
      taban : (x, y) nesnenin AYAK/alt orta noktasi (resim koordinati)
      oran  : nesne yuksekliginin resim yuksekligine orani (boyut)
      uyum  : renk/isik uyumu gucu 0-1
      golge : zemine yumusak golge ekle, golge_yon -1..1 (isik yonu)
      poisson: cv2.seamlessClone ile dikissiz karistir
    """
    H, W = bg.shape[:2]
    oh = max(8, int(H * oran))
    ow = max(8, int(obj.shape[1] * oh / obj.shape[0]))
    interp = cv2.INTER_AREA if oh < obj.shape[0] else cv2.INTER_CUBIC
    o = cv2.resize(obj, (ow, oh), interpolation=interp)
    a = cv2.resize(alfa, (ow, oh), interpolation=interp)
    if yansit:
        o, a = o[:, ::-1].copy(), a[:, ::-1].copy()

    bx, by = taban
    x0, y0 = int(bx - ow / 2), int(by - oh)

    # renk / isik uyumu: nesnenin cevresindeki bolgenin istatistigine gore
    gx0, gx1 = max(x0 - ow // 2, 0), min(x0 + ow + ow // 2, W)
    gy0, gy1 = max(y0 - oh // 2, 0), min(y0 + oh + oh // 2, H)
    hedef = bg[gy0:gy1, gx0:gx1]
    if hedef.size == 0:
        hedef = bg
    o = renk_aktar(o, a, hedef, uyum)

    cikti = bg.copy()

    # resim disina tasan kismi kirp
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1, sy1 = min(ow, W - x0), min(oh, H - y0)
    if sx1 <= sx0 or sy1 <= sy0:
        return cikti
    dx0, dy0 = x0 + sx0, y0 + sy0
    dx1, dy1 = x0 + sx1, y0 + sy1
    o_k, a_k = o[sy0:sy1, sx0:sx1], a[sy0:sy1, sx0:sx1]

    # golge: nesne siluetini zemine yatirip (egme) bulaniklastir
    if golge:
        tam = np.zeros((H, W), np.float32)
        tam[dy0:dy1, dx0:dx1] = a_k.astype(np.float32) / 255.0
        kx, ky = golge_yon * 0.8, 0.3
        M = np.float32([[1, -kx, kx * by], [0, ky, (1 - ky) * by]])
        g = cv2.warpAffine(tam, M, (W, H))
        g = cv2.GaussianBlur(g, (0, 0), max(1.0, oh * 0.02)) * 0.5
        cikti = (cikti.astype(np.float32) * (1 - g[..., None])).astype(np.uint8)

    # (seamlessClone)
    if poisson:
        pad = 2
        m = (a_k > 127).astype(np.uint8) * 255
        m = cv2.erode(m, np.ones((3, 3), np.uint8))
        kaynak = cv2.copyMakeBorder(o_k, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        maske = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        merkez = (dx0 + (dx1 - dx0) // 2, dy0 + (dy1 - dy0) // 2)
        sw, sh = kaynak.shape[1], kaynak.shape[0]
        sigiyor = (merkez[0] - sw // 2 >= 0 and merkez[1] - sh // 2 >= 0 and
                   merkez[0] + sw - sw // 2 <= W and merkez[1] + sh - sh // 2 <= H)
        if sigiyor and m.any():
            try:
                return cv2.seamlessClone(kaynak, cikti, maske, merkez, cv2.NORMAL_CLONE)
            except cv2.error:
                pass                                   # olmazsa alfa karistirmaya don

    # --- alfa karistirma (kenar yumusatmali)
    af = cv2.erode(a_k, np.ones((3, 3), np.uint8))
    af = cv2.GaussianBlur(af, (0, 0), 1.2).astype(np.float32)[..., None] / 255.0
    roi = cikti[dy0:dy1, dx0:dx1].astype(np.float32)
    cikti[dy0:dy1, dx0:dx1] = (roi * (1 - af) + o_k.astype(np.float32) * af).astype(np.uint8)
    return cikti


class EklePenceresi:
    """Nesneyi yerlestirme penceresi: tikla = konum, kaydiricilar = boyut/uyum/golge."""

    def __init__(self, ana, obj, alfa):
        self.ana, self.obj, self.alfa = ana, obj, alfa
        self.bg = ana.calisma
        H, W = self.bg.shape[:2]
        self.taban = [W / 2, H * 0.85]
        self.olcek = 1.0
        self._tk = None
        self._bekliyor = False

        self.pen = tk.Toplevel(ana.kok)
        self.pen.title("Nesne Ekle  -  konum için resme tıklayın / sürükleyin")

        panel = ttk.Frame(self.pen, padding=8)
        panel.pack(side="left", fill="y")

        self.boyut = tk.DoubleVar(value=30)
        self.uyum = tk.DoubleVar(value=70)
        self.yon = tk.DoubleVar(value=0.5)
        self.golge = tk.BooleanVar(value=True)
        self.yansit = tk.BooleanVar(value=False)
        self.poisson = tk.BooleanVar(value=False)

        def kaydirici(metin, degisken, a, b):
            ttk.Label(panel, text=metin).pack(anchor="w", pady=(8, 0))
            ttk.Scale(panel, from_=a, to=b, variable=degisken, length=200).pack(fill="x")

        kaydirici("Boyut (resim yüksekliğinin %'si)", self.boyut, 3, 100)
        kaydirici("Renk / ışık uyumu (%)", self.uyum, 0, 100)
        kaydirici("Gölge yönü (-1 sol ... +1 sağ)", self.yon, -1, 1)
        ttk.Checkbutton(panel, text="Gölge ekle", variable=self.golge).pack(anchor="w", pady=(8, 0))
        ttk.Checkbutton(panel, text="Yatay çevir", variable=self.yansit).pack(anchor="w")
        ttk.Checkbutton(panel, text="Poisson karıştırma (Uygula'da işlenir)",
                        variable=self.poisson).pack(anchor="w")
        ttk.Label(panel, text="Tıklanan nokta nesnenin\nalt-orta noktası (ayağı) olur.",
                  foreground="gray").pack(anchor="w", pady=8)
        ttk.Button(panel, text="Uygula", command=self.uygula).pack(fill="x", pady=2)
        ttk.Button(panel, text="İptal", command=self.pen.destroy).pack(fill="x", pady=2)

        self.etiket = ttk.Label(self.pen)
        self.etiket.pack(side="left", padx=6, pady=6)
        self.etiket.bind("<Button-1>", self.tikla)
        self.etiket.bind("<B1-Motion>", self.tikla)

        for v in (self.boyut, self.uyum, self.yon, self.golge, self.yansit):
            v.trace_add("write", lambda *a: self.yenile())
        self.yenile()

    def render(self, poisson):
        return birlestir(self.bg, self.obj, self.alfa, tuple(self.taban),
                         self.boyut.get() / 100, self.yansit.get(),
                         self.uyum.get() / 100, self.golge.get(), self.yon.get(), poisson)

    def yenile(self):
        if self._bekliyor:
            return
        self._bekliyor = True
        self.pen.after(30, self._ciz)

    def _ciz(self):
        self._bekliyor = False
        self._tk, self.olcek = tk_resim(self.render(False))   # onizleme hizli: Poisson yok
        self.etiket.config(image=self._tk)

    def tikla(self, olay):
        self.taban = [olay.x / self.olcek, olay.y / self.olcek]
        self.yenile()

    def uygula(self):
        sonuc = self.render(self.poisson.get())
        self.ana.gecmise_ekle()
        self.ana.calisma = sonuc
        self.ana.goster(sonuc)
        self.ana.durum.config(text="Nesne eklendi. İstersen 'Geri Al' ile geri alabilirsin.")
        self.pen.destroy()


#  UYGULAMA
class Uygulama:
    def __init__(self, kok):
        self.kok = kok
        kok.title("Proje-I: Görüntü İşleme Uygulaması")
        self.img = None            # orijinal BGR resim
        self.calisma = None        # uzerinde calisilan resim
        self.nesneler = []         # [{"ad":..., "maske":...}, ...]
        self.gecmis = []           # geri al icin
        self.model = None
        self._tk_img = None
        self.secili_renk = None
        self.cap = None
        self.video_calisiyor = False

        self._arayuz_kur()
        self._model_yukle()
        kok.protocol("WM_DELETE_WINDOW", self.kapat)

    #  arayuz
    def _arayuz_kur(self):
        sol = ttk.Frame(self.kok, padding=6)
        sol.pack(side="left", fill="y")
        orta = ttk.Frame(self.kok, padding=6)
        orta.pack(side="left", fill="y")
        self.etiket = ttk.Label(self.kok)
        self.etiket.pack(side="left", expand=True, padx=8, pady=8)

        #  sol: Kisim 1
        ttk.Button(sol, text="Resim Aç", command=self.resim_ac).pack(fill="x", pady=2)
        ttk.Button(sol, text="Nesneleri Tespit Et", command=self.tespit_et).pack(fill="x", pady=2)

        ttk.Label(sol, text="Güven eşiği:").pack(anchor="w", pady=(8, 0))
        self.guven = tk.DoubleVar(value=0.25)
        ttk.Scale(sol, from_=0.05, to=0.9, variable=self.guven).pack(fill="x")
        self.guven_etiket = ttk.Label(sol, text="0.25")
        self.guven_etiket.pack(anchor="w")
        self.guven.trace_add("write", lambda *a: self.guven_etiket.config(text=f"{self.guven.get():.2f}"))

        ttk.Label(sol, text="Bulunan nesneler:").pack(anchor="w", pady=(8, 0))
        self.liste = tk.Listbox(sol, width=26, height=14, exportselection=False)
        self.liste.pack(fill="y", pady=2)
        self.liste.bind("<<ListboxSelect>>", self.secim_degisti)

        ttk.Button(sol, text="Manuel Nesne Ekle (GrabCut)", command=self.manuel_ekle).pack(fill="x", pady=(6, 2))
        ttk.Button(sol, text="Konturu Temizle", command=self.goster_calisma).pack(fill="x", pady=2)
        ttk.Button(sol, text="Geri Al", command=self.geri_al).pack(fill="x", pady=2)

        self.durum = ttk.Label(sol, text="Hazır.", wraplength=190, foreground="gray")
        self.durum.pack(anchor="w", pady=(8, 0))

        # ---- orta: Kisim 2
        k2 = ttk.LabelFrame(orta, text="Kısım 2: Nesneyi Çıkar", padding=6)
        k2.pack(fill="x", pady=4)
        ttk.Label(k2, text="Doldurma yöntemi:").pack(anchor="w")
        self.yontem_var = tk.StringVar(value=list(YONTEMLER)[0])
        ttk.Combobox(k2, textvariable=self.yontem_var, values=list(YONTEMLER),
                     state="readonly", width=26).pack(fill="x")
        ttk.Label(k2, text="Genişletme (px):").pack(anchor="w", pady=(6, 0))
        self.genislet = tk.DoubleVar(value=9)
        ttk.Scale(k2, from_=1, to=40, variable=self.genislet).pack(fill="x")
        self.renk_kutu = tk.Label(k2, text="Renk seç", bg="#888888", fg="white", cursor="hand2")
        self.renk_kutu.pack(fill="x", pady=4)
        self.renk_kutu.bind("<Button-1>", lambda e: self.renk_sec())
        ttk.Button(k2, text="Seçili Nesneyi Çıkar", command=self.cikar).pack(fill="x", pady=2)
        ttk.Button(k2, text="Farklı Kaydet...", command=self.farkli_kaydet).pack(fill="x", pady=2)

        #  orta: Kisim 3
        k3 = ttk.LabelFrame(orta, text="Kısım 3: Dışarıdan Nesne Ekle", padding=6)
        k3.pack(fill="x", pady=4)
        ttk.Button(k3, text="Nesne Resmi Seç ve Ekle...", command=self.nesne_ekle).pack(fill="x", pady=2)
        ttk.Label(k3, text="Şeffaf PNG en iyi sonucu verir.\nDeğilse kutu çizip GrabCut ile ayrılır.",
                  foreground="gray").pack(anchor="w")

        # orta: Bonus
        bn = ttk.LabelFrame(orta, text="Bonus: Video / Kamera", padding=6)
        bn.pack(fill="x", pady=4)
        ttk.Button(bn, text="Video Aç...", command=self.video_ac).pack(fill="x", pady=2)
        ttk.Button(bn, text="Kamerayı Başlat", command=self.kamera_ac).pack(fill="x", pady=2)
        ttk.Button(bn, text="Durdur", command=self.video_durdur).pack(fill="x", pady=2)

    def _model_yukle(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(MODEL_ADI)
            self.durum.config(text="Model yüklendi.")
        except Exception as e:
            self.durum.config(text="Model yüklenemedi. Manuel (GrabCut) mod kullanılabilir.")
            messagebox.showwarning("Model", f"YOLO modeli yüklenemedi:\n{e}")

    # gosterim
    def goster(self, bgr):
        self._tk_img, _ = tk_resim(bgr)
        self.etiket.config(image=self._tk_img)

    def goster_calisma(self):
        if self.calisma is not None:
            self.liste.selection_clear(0, "end")
            self.goster(self.calisma)

    def liste_yenile(self):
        self.liste.delete(0, "end")
        for n in self.nesneler:
            self.liste.insert("end", n["ad"])

    # geri al
    def gecmise_ekle(self):
        self.gecmis.append((self.calisma.copy(), list(self.nesneler)))
        if len(self.gecmis) > 10:
            self.gecmis.pop(0)

    def geri_al(self):
        if not self.gecmis:
            self.durum.config(text="Geri alınacak işlem yok.")
            return
        self.calisma, self.nesneler = self.gecmis.pop()
        self.liste_yenile()
        self.goster(self.calisma)
        self.durum.config(text="Son işlem geri alındı.")

    #  KISIM 1
    def resim_ac(self):
        yol = filedialog.askopenfilename(
            filetypes=[("Resimler", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Hepsi", "*.*")])
        if not yol:
            return
        try:
            img = resim_oku(yol)
        except Exception as e:
            messagebox.showerror("Hata", f"Resim açılamadı:\n{e}")
            return
        self.video_durdur(goster=False)
        self.img = img
        self.calisma = img.copy()
        self.nesneler = []
        self.gecmis = []
        self.liste_yenile()
        self.goster(self.calisma)
        self.durum.config(text=f"Resim yüklendi ({img.shape[1]}x{img.shape[0]}).")

    def tespit_et(self):
        if self.calisma is None:
            messagebox.showinfo("Bilgi", "Önce bir resim açın.")
            return
        if self.model is None:
            messagebox.showinfo("Bilgi", "Model yok. Manuel (GrabCut) ekleme kullanın.")
            return

        self.durum.config(text="Tespit yapılıyor...")
        self.kok.update_idletasks()
        sonuc = self.model(self.calisma, conf=self.guven.get(), verbose=False)[0]

        self.nesneler = []
        if sonuc.masks is None or len(sonuc.masks.xy) == 0:
            self.liste_yenile()
            self.durum.config(text="Nesne bulunamadı. Eşiği düşürün veya manuel ekleyin.")
            self.goster(self.calisma)
            return

        h, w = self.calisma.shape[:2]
        adlar = [sonuc.names[int(c)] for c in sonuc.boxes.cls]
        for i, poligon in enumerate(sonuc.masks.xy):
            if len(poligon) < 3:
                continue
            maske = np.zeros((h, w), np.uint8)
            cv2.fillPoly(maske, [poligon.astype(np.int32)], 255)
            self.nesneler.append({"ad": adlar[i], "maske": maske})

        self._numarala()
        self.liste_yenile()
        self.goster(self.calisma)
        self.durum.config(text=f"{len(self.nesneler)} nesne bulundu.")

    def _numarala(self):
        """Ayni adli nesnelere 'house #1', 'house #2' gibi numara ekler."""
        temel = [n["ad"].split(" #")[0] for n in self.nesneler]
        sayac = {}
        for n, t in zip(self.nesneler, temel):
            if temel.count(t) > 1:
                sayac[t] = sayac.get(t, 0) + 1
                n["ad"] = f"{t} #{sayac[t]}"
            else:
                n["ad"] = t

    def secim_degisti(self, _olay=None):
        sec = self.liste.curselection()
        if not sec or self.calisma is None:
            return
        self.goster(kontur_ciz(self.calisma, self.nesneler[sec[0]]["maske"]))

    def manuel_ekle(self):
        if self.calisma is None:
            messagebox.showinfo("Bilgi", "Önce bir resim açın.")
            return
        # Fareyle kutu cizilir: ENTER/SPACE onayla, C iptal
        kutu = cv2.selectROI("Nesneyi cevreleyen kutu cizin (ENTER = onay)", self.calisma, False)
        cv2.destroyAllWindows()
        if kutu[2] < 5 or kutu[3] < 5:
            return
        ad = simpledialog.askstring("Nesne adı", "Bu nesneye ne ad verelim? (örnek: göl)")
        if not ad:
            return
        self.durum.config(text="GrabCut çalışıyor...")
        self.kok.update_idletasks()
        maske = grabcut_maske(self.calisma, kutu)
        if cv2.countNonZero(maske) == 0:
            messagebox.showwarning("Uyarı", "Nesne ayrılamadı, kutuyu değiştirip tekrar deneyin.")
            return
        self.nesneler.append({"ad": ad, "maske": maske})
        self.liste_yenile()
        self.liste.selection_clear(0, "end")
        self.liste.selection_set("end")
        self.secim_degisti()
        self.durum.config(text=f"'{ad}' eklendi.")

    #  KISIM 2
    def renk_sec(self):
        c = colorchooser.askcolor(title="Doldurma rengi")
        if c and c[0]:
            r, g, b = [int(v) for v in c[0]]
            self.secili_renk = (b, g, r)                 # OpenCV BGR
            self.renk_kutu.config(bg=c[1], text="Seçilen renk")
            self.yontem_var.set([k for k, v in YONTEMLER.items() if v == "renk"][0])

    def cikar(self):
        if self.calisma is None:
            messagebox.showinfo("Bilgi", "Önce bir resim açın.")
            return
        sec = self.liste.curselection()
        if not sec:
            messagebox.showinfo("Bilgi", "Listeden çıkarılacak nesneyi seçin.")
            return
        yontem = YONTEMLER[self.yontem_var.get()]
        if yontem == "renk" and self.secili_renk is None:
            self.renk_sec()
            if self.secili_renk is None:
                return
        nesne = self.nesneler[sec[0]]
        self.durum.config(text="Nesne çıkarılıyor...")
        self.kok.update_idletasks()
        try:
            self.gecmise_ekle()
            self.calisma = nesne_cikar(self.calisma, nesne["maske"], yontem,
                                       int(self.genislet.get()), self.secili_renk)
        except Exception as e:
            self.gecmis.pop()
            messagebox.showerror("Hata", f"Çıkarma başarısız:\n{e}")
            return
        del self.nesneler[sec[0]]
        self.liste_yenile()
        self.goster(self.calisma)
        self.durum.config(text=f"'{nesne['ad']}' çıkarıldı. 'Farklı Kaydet' ile kaydedebilirsin.")

    def farkli_kaydet(self):
        if self.calisma is None:
            messagebox.showinfo("Bilgi", "Kaydedilecek resim yok.")
            return
        yol = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp"), ("TIFF", "*.tif")])
        if not yol:
            return
        uzanti = os.path.splitext(yol)[1] or ".png"
        ok, tampon = cv2.imencode(uzanti, self.calisma)
        if not ok:
            messagebox.showerror("Hata", "Bu biçimde kaydedilemedi.")
            return
        tampon.tofile(yol)                                # Turkce yollarda da calisir
        self.durum.config(text=f"Kaydedildi: {os.path.basename(yol)}")

    # KISIM 3
    def nesne_ekle(self):
        if self.calisma is None:
            messagebox.showinfo("Bilgi", "Önce bir resim açın.")
            return
        yol = filedialog.askopenfilename(
            title="Eklenecek nesnenin resmi",
            filetypes=[("Resimler", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Hepsi", "*.*")])
        if not yol:
            return
        try:
            obj, alfa = nesne_hazirla(yol)
        except Exception as e:
            messagebox.showerror("Hata", f"Nesne hazırlanamadı:\n{e}")
            return
        EklePenceresi(self, obj, alfa)

    # ------------------------------------------------------------ BONUS: video
    def video_ac(self):
        yol = filedialog.askopenfilename(
            filetypes=[("Videolar", "*.avi *.mp4 *.mov *.mkv *.wmv"), ("Hepsi", "*.*")])
        if yol:
            self.video_baslat(yol)

    def kamera_ac(self):
        self.video_baslat(0)

    def video_baslat(self, kaynak):
        if self.model is None:
            messagebox.showinfo("Bilgi", "Video tespiti için model gerekli.")
            return
        self.video_durdur(goster=False)
        self.cap = cv2.VideoCapture(kaynak, cv2.CAP_DSHOW) if kaynak == 0 else cv2.VideoCapture(kaynak)
        if not self.cap.isOpened():
            messagebox.showerror("Hata", "Video/kamera açılamadı.")
            self.cap = None
            return
        self.video_calisiyor = True
        self.durum.config(text="Video çalışıyor. 'Durdur' ile bitir.")
        self._video_dongu()

    def _video_dongu(self):
        if not self.video_calisiyor or self.cap is None:
            return
        ok, kare = self.cap.read()
        if not ok:
            self.video_durdur()
            return
        sonuc = self.model(kare, conf=self.guven.get(), imgsz=480, verbose=False)[0]
        self.goster(sonuc.plot())                         # kutu + maske + etiket cizili kare
        self.kok.after(1, self._video_dongu)

    def video_durdur(self, goster=True):
        self.video_calisiyor = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if goster and self.calisma is not None:
            self.goster(self.calisma)
        if goster:
            self.durum.config(text="Video durduruldu.")

    def kapat(self):
        self.video_durdur(goster=False)
        self.kok.destroy()


if __name__ == "__main__":
    kok = tk.Tk()
    Uygulama(kok)
    kok.mainloop()
