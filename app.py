import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Deteksi Penyakit Daun",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #16a34a;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-header {
        text-align: center;
        color: #374151;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    .status-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.8em;
        font-weight: bold;
        margin: 20px 0;
    }
    .status-sehat {
        background-color: #4caf50;
        color: white;
    }
    .status-ringan {
        background-color: #ff9800;
        color: white;
    }
    .status-sedang {
        background-color: #ff5722;
        color: white;
    }
    .status-parah {
        background-color: #d32f2f;
        color: white;
    }
    .metric-box {
        padding: 15px;
        border-radius: 8px;
        background: linear-gradient(135deg, #10b981 0%, #22c55e 100%);
        color: #ffffff;
        text-align: center;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        margin-top: 5px;
    }
    .footer {
        text-align: center;
        padding: 20px;
        background-color: #ffffff;
        color: #374151;
        border-radius: 10px;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

def buat_citra_dummy(kondisi="sehat"):
    """Membuat gambar simulasi daun dengan/tanpa bercak penyakit"""
    h, w = 400, 400
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Buat bentuk daun (elips)
    center_y, center_x = 200, 200
    axes_major, axes_minor = 150, 100
    
    # Buat mask daun
    y, x = np.ogrid[:h, :w]
    mask = ((x - center_x) ** 2 / axes_major ** 2 + 
            (y - center_y) ** 2 / axes_minor ** 2) <= 1
    
    if kondisi == "sehat":
        # Daun sehat: hijau cerah
        img[mask] = [34, 139, 34]  # RGB untuk hijau
        
    elif kondisi == "ringan":
        # Daun dengan bercak ringan
        img[mask] = [34, 139, 34]
        
        # Tambahkan beberapa bercak kuning kecil
        for _ in range(5):
            bx, by = np.random.randint(100, 300, 2)
            br = np.random.randint(8, 15)
            bmask = (x - bx) ** 2 + (y - by) ** 2 <= br ** 2
            img[bmask & mask] = [139, 139, 0]  # Kuning gelap
            
    elif kondisi == "sedang":
        # Daun dengan bercak sedang
        img[mask] = [34, 139, 34]
        
        # Tambahkan bercak coklat dan kuning
        for _ in range(10):
            bx, by = np.random.randint(80, 320, 2)
            br = np.random.randint(10, 20)
            bmask = (x - bx) ** 2 + (y - by) ** 2 <= br ** 2
            if np.random.random() > 0.5:
                img[bmask & mask] = [139, 90, 0]  # Coklat
            else:
                img[bmask & mask] = [139, 139, 0]  # Kuning gelap
                
    else:  # parah
        # Daun dengan bercak parah
        img[mask] = [34, 100, 34]  # Hijau lebih gelap
        
        # Tambahkan banyak bercak coklat dan kuning
        for _ in range(20):
            bx, by = np.random.randint(70, 330, 2)
            br = np.random.randint(12, 30)
            bmask = (x - bx) ** 2 + (y - by) ** 2 <= br ** 2
            if np.random.random() > 0.3:
                img[bmask & mask] = [90, 60, 20]  # Coklat gelap
            else:
                img[bmask & mask] = [139, 120, 0]  # Kuning coklat
    
    # Tambahkan noise untuk realisme
    noise = np.random.randint(-15, 15, (h, w, 3))
    final_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return final_img

def deteksi_bercak_penyakit(img_rgb, warna_dasar="Hijau (Default)", sensitivitas=5):
    """Deteksi bercak penyakit pada daun berdasarkan analisis warna dan tekstur"""
    # Konversi ke HSV dan LAB
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    
    # Sesuaikan threshold berdasarkan warna dasar
    if warna_dasar == "Hijau (Default)":
        # Deteksi area hijau sehat
        lower_healthy = np.array([35, 40, 20])
        upper_healthy = np.array([85, 255, 255])
        mask_healthy = cv2.inRange(hsv, lower_healthy, upper_healthy)
        
        # Bercak = coklat + kuning gelap
        lower_brown = np.array([10, 50, 20])
        upper_brown = np.array([30, 255, 150])
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        
        lower_yellow = np.array([20, 50, 50])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        mask_disease = cv2.bitwise_or(mask_brown, mask_yellow)
        
    elif warna_dasar == "Kuning/Keemasan":
        # Untuk daun kuning, fokus ke area yang LEBIH GELAP atau COKLAT
        lower_healthy = np.array([20, 30, 80])  # Kuning cerah
        upper_healthy = np.array([40, 255, 255])
        mask_healthy = cv2.inRange(hsv, lower_healthy, upper_healthy)
        
        # Hanya deteksi coklat gelap dan area yang sangat berbeda
        lower_brown = np.array([10, 40, 20])
        upper_brown = np.array([25, 255, 120])  # Lebih ketat
        mask_disease = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Tambahan: deteksi area gelap dengan LAB
        l_channel = lab[:, :, 0]
        _, dark_spots = cv2.threshold(l_channel, 80, 255, cv2.THRESH_BINARY_INV)
        mask_disease = cv2.bitwise_or(mask_disease, dark_spots)
        
    elif warna_dasar == "Kemerahan/Ungu":
        # Untuk daun merah/ungu
        lower_healthy1 = np.array([0, 30, 30])
        upper_healthy1 = np.array([10, 255, 255])
        lower_healthy2 = np.array([140, 30, 30])
        upper_healthy2 = np.array([180, 255, 255])
        mask_healthy = cv2.bitwise_or(
            cv2.inRange(hsv, lower_healthy1, upper_healthy1),
            cv2.inRange(hsv, lower_healthy2, upper_healthy2)
        )
        
        # Deteksi coklat dan hitam
        lower_brown = np.array([10, 30, 20])
        upper_brown = np.array([30, 255, 100])
        mask_disease = cv2.inRange(hsv, lower_brown, upper_brown)
        
    else:  # Custom - gunakan analisis tekstur
        # Deteksi semua area non-background
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        _, mask_healthy = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        
        # Gunakan variance untuk deteksi bercak (area tidak seragam)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        variance = cv2.absdiff(gray, blur)
        _, mask_disease = cv2.threshold(variance, 20, 255, cv2.THRESH_BINARY)
        
        # Morfologi untuk hapus noise kecil
        kernel = np.ones((5, 5), np.uint8)
        mask_disease = cv2.morphologyEx(mask_disease, cv2.MORPH_OPEN, kernel)
    
    # Sesuaikan sensitivitas dengan morfologi
    sens_factor = sensitivitas / 5.0  # Normalisasi ke 0.2 - 2.0
    kernel_size = max(3, int(7 - sens_factor * 2))  # Kernel lebih kecil = lebih sensitif
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    if sensitivitas > 5:
        # Lebih sensitif: kurangi noise removal
        mask_disease = cv2.morphologyEx(mask_disease, cv2.MORPH_CLOSE, kernel, iterations=1)
    else:
        # Kurang sensitif: lebih banyak noise removal
        mask_disease = cv2.morphologyEx(mask_disease, cv2.MORPH_OPEN, kernel, iterations=2)
        mask_disease = cv2.morphologyEx(mask_disease, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Deteksi pola: bercak penyakit = spot tidak merata
    # Hitung connected components untuk analisis pola
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_disease, connectivity=8)
    
    # Filter komponen yang terlalu kecil (noise) atau terlalu besar (false positive)
    mask_filtered = np.zeros_like(mask_disease)
    total_area = img_rgb.shape[0] * img_rgb.shape[1]
    min_area = total_area * 0.0005  # 0.05% dari total
    max_area = total_area * 0.3     # 30% dari total
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area < area < max_area:
            mask_filtered[labels == i] = 255
    
    mask_disease = mask_filtered
    mask_green = mask_healthy
    
    # Hitung persentase
    total_leaf_pixels = np.sum(mask_green > 0)
    disease_pixels = np.sum(mask_disease > 0)
    
    if total_leaf_pixels > 0:
        persentase_penyakit = (disease_pixels / total_leaf_pixels) * 100
    else:
        persentase_penyakit = 0
    
    # Tentukan tingkat kesehatan
    if persentase_penyakit < 5:
        tingkat = "SEHAT"
        status_class = "status-sehat"
        keterangan = "Daun dalam kondisi baik"
    elif persentase_penyakit < 15:
        tingkat = "TERINFEKSI RINGAN"
        status_class = "status-ringan"
        keterangan = "Bercak penyakit mulai muncul"
    elif persentase_penyakit < 30:
        tingkat = "TERINFEKSI SEDANG"
        status_class = "status-sedang"
        keterangan = "Perlu penanganan segera"
    else:
        tingkat = "TERINFEKSI PARAH"
        status_class = "status-parah"
        keterangan = "Kondisi kritis, butuh treatment intensif"
    
    return {
        'tingkat': tingkat,
        'status_class': status_class,
        'persentase_penyakit': persentase_penyakit,
        'keterangan': keterangan,
        'mask_green': mask_green,
        'mask_disease': mask_disease,
        'luas_daun': total_leaf_pixels,
        'luas_bercak': disease_pixels
    }

def proses_citra(img_array, warna_dasar="Hijau (Default)", sensitivitas=5):
    """Fungsi utama untuk memproses citra daun"""
    # Simpan gambar RGB original
    if len(img_array.shape) == 3:
        img_rgb = img_array.copy()
        gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = img_array
        img_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    
    # Deteksi bercak penyakit dengan parameter kalibrasi
    hasil_penyakit = deteksi_bercak_penyakit(img_rgb, warna_dasar, sensitivitas)
    
    # 1. Enhancement: Median Filtering
    median_filtered = cv2.medianBlur(gray_img, 5)
    
    # 2. Segmentasi: Otsu's Thresholding
    _, binary_img = cv2.threshold(median_filtered, 0, 255, 
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Morfologi: Closing untuk mengisi lubang
    kernel = np.ones((7, 7), np.uint8)
    morph_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    
    # 4. Edge Detection
    edges = cv2.Canny(morph_img, 50, 150)
    
    # 5. Ekstraksi Kontur
    contours, _ = cv2.findContours(morph_img, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
    
    # 6. Visualisasi hasil
    output_img = img_rgb.copy()
    segmented_img = img_rgb.copy()
    
    luas_daun = 0
    if len(contours) > 0:
        # Ambil kontur terbesar (daun)
        cnt = max(contours, key=cv2.contourArea)
        luas_daun = cv2.contourArea(cnt)
        
        # Gambar kontur daun (hijau)
        cv2.drawContours(output_img, [cnt], -1, (0, 255, 0), 3)
        
        # Buat mask untuk segmentasi daun
        mask = np.zeros(gray_img.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        
        # Terapkan mask ke gambar original
        segmented_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
        
        # Overlay bercak penyakit dengan warna merah
        disease_mask_3ch = cv2.cvtColor(hasil_penyakit['mask_disease'], 
                                        cv2.COLOR_GRAY2RGB)
        red_overlay = np.zeros_like(img_rgb)
        red_overlay[:, :, 0] = hasil_penyakit['mask_disease']  # Channel merah
        output_img = cv2.addWeighted(output_img, 1, red_overlay, 0.5, 0)
    
    return {
        'original': img_rgb,
        'gray': gray_img,
        'filtered': median_filtered,
        'binary': binary_img,
        'morph': morph_img,
        'edges': edges,
        'segmented': segmented_img,
        'result': output_img,
        'luas_daun': luas_daun,
        'penyakit_info': hasil_penyakit
    }

# Header
col_header1, col_header2, col_header3 = st.columns([1, 2, 1])
with col_header2:
    st.markdown("<div class='main-header'>🍃 Sistem Deteksi Penyakit Daun</div>", 
                unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Segmentasi Daun & Analisis Bercak Penyakit dengan Computer Vision</div>", 
                unsafe_allow_html=True)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=100)
    st.header("⚙️ Pengaturan")
    
    st.markdown("### Mode Input")
    st.markdown("📷 Upload Gambar")
    
    st.markdown("---")
    st.markdown("### 🎨 Kalibrasi Warna Dasar")
    warna_dasar = st.selectbox(
        "Warna Dasar Daun:",
        ["Hijau (Default)", "Kuning/Keemasan", "Kemerahan/Ungu", "Custom"],
        help="Pilih warna dasar alami tanaman untuk menghindari false positive"
    )
    
    if warna_dasar == "Custom":
        st.info("💡 Mode Custom: Sistem akan menganalisis distribusi warna untuk deteksi bercak")
    
    sensitivitas = st.slider(
        "Sensitivitas Deteksi:",
        min_value=1,
        max_value=10,
        value=5,
        help="Tingkatkan untuk deteksi lebih sensitif, turunkan untuk mengurangi false positive"
    )
    
    st.markdown("---")
    st.markdown("### 🏥 Kriteria Kesehatan")
    st.markdown("""
    **Tingkat Infeksi:**
    - ✅ **Sehat**: Bercak < 5%
    - ⚠️ **Ringan**: Bercak 5-15%
    - 🔶 **Sedang**: Bercak 15-30%
    - ❌ **Parah**: Bercak > 30%
    
    **Indikator Penyakit:**
    - 🟤 Bercak coklat gelap
    - ⚫ Spot hitam/nekrosis
    - 📊 Pola tidak merata
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Tips Penggunaan")
    st.markdown("""
    **Untuk daun berwarna kuning alami:**
    - Pilih "Kuning/Keemasan"
    - Sistem akan fokus ke bercak coklat
    
    **Untuk daun merah/ungu:**
    - Pilih "Kemerahan/Ungu"
    - Deteksi disesuaikan untuk warna ini
    
    **Mode Custom:**
    - Menggunakan analisis tekstur
    - Cocok untuk warna tidak standar
    
    **Sensitivitas:**
    - Rendah (1-3): Lebih konservatif
    - Sedang (4-6): Balanced
    - Tinggi (7-10): Lebih sensitif
    """)
    
    st.markdown("---")
    st.markdown("### 🔬 Tahapan Proses")
    st.markdown("""
    1. **Kalibrasi Warna**: Sesuaikan threshold
    2. **Enhancement**: Median Filter
    3. **Segmentasi**: Otsu Thresholding
    4. **Analisis Tekstur**: Deteksi pola
    5. **Filter Noise**: Connected Components
    6. **Klasifikasi**: Multi-criteria
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Input Gambar Daun")
    
    uploaded_file = st.file_uploader(
        "Upload gambar daun tanaman",
        type=['png', 'jpg', 'jpeg'],
        help="Upload gambar daun yang akan dianalisis"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        st.image(image, caption="Gambar yang diupload", 
                use_container_width=True)
    else:
        st.info("👆 Silakan upload gambar daun terlebih dahulu")
        img_array = None

with col2:
    st.subheader("🎯 Hasil Analisis")
    
    if img_array is not None:
        with st.spinner("⏳ Menganalisis kondisi daun..."):
            hasil = proses_citra(img_array, warna_dasar, sensitivitas)
            
            # Tampilkan status kesehatan
            info = hasil['penyakit_info']
            st.markdown(f"""
            <div class='status-box {info['status_class']}'>
                🍃 {info['tingkat']}
            </div>
            """, unsafe_allow_html=True)
            
            # Info kalibrasi
            if warna_dasar != "Hijau (Default)":
                st.info(f"🎨 **Mode Kalibrasi**: {warna_dasar} | Sensitivitas: {sensitivitas}/10")
            
            st.info(f"💡 **Keterangan**: {info['keterangan']}")
            
            # Metrik
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div class='metric-box'>
                    <div>🔬 Persentase Bercak</div>
                    <div class='metric-value'>{info['persentase_penyakit']:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f"""
                <div class='metric-box'>
                    <div>📐 Luas Daun</div>
                    <div class='metric-value'>{hasil['luas_daun']:,} px²</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Metrik tambahan
            col_c, col_d = st.columns(2)
            with col_c:
                st.metric("🟤 Luas Bercak", f"{info['luas_bercak']:,} px²")
            with col_d:
                kesehatan = 100 - info['persentase_penyakit']
                st.metric("💚 Tingkat Kesehatan", f"{kesehatan:.1f}%")
            
            # Gambar hasil
            st.image(hasil['result'], 
                    caption="Hasil Segmentasi (Hijau: Kontur Daun, Merah: Area Bercak)", 
                    use_container_width=True, clamp=True)
    else:
        st.info("⬅️ Silakan input gambar daun terlebih dahulu")

# Tampilkan tahapan proses
if img_array is not None:
    st.markdown("---")
    st.subheader("🔍 Tahapan Pemrosesan Detail")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1️⃣ Original", "2️⃣ Filtered", "3️⃣ Segmentasi", 
        "4️⃣ Edges", "5️⃣ Hasil Akhir"
    ])
    
    with tab1:
        col_t1a, col_t1b = st.columns(2)
        with col_t1a:
            st.image(hasil['original'], caption="Gambar Original (RGB)", 
                    use_container_width=True, clamp=True)
        with col_t1b:
            st.image(hasil['gray'], caption="Grayscale", 
                    use_container_width=True, clamp=True)
        st.markdown("**Deskripsi**: Gambar input asli dan konversi ke grayscale untuk processing")
    
    with tab2:
        st.image(hasil['filtered'], caption="Setelah Median Filter", 
                use_container_width=True, clamp=True)
        st.markdown("**Deskripsi**: Noise reduction dengan mempertahankan detail tepi daun")
    
    with tab3:
        col_t3a, col_t3b = st.columns(2)
        with col_t3a:
            st.image(hasil['binary'], caption="Binary Thresholding", 
                    use_container_width=True, clamp=True)
        with col_t3b:
            st.image(hasil['segmented'], caption="Daun Tersegmentasi", 
                    use_container_width=True, clamp=True)
        st.markdown("**Deskripsi**: Pemisahan objek daun dari background")
    
    with tab4:
        st.image(hasil['edges'], caption="Deteksi Tepi (Canny)", 
                use_container_width=True, clamp=True)
        st.markdown("**Deskripsi**: Tepi daun terdeteksi untuk analisis bentuk dan boundary")
    
    with tab5:
        col_t5a, col_t5b = st.columns(2)
        with col_t5a:
            # Visualisasi mask penyakit
            disease_viz = cv2.cvtColor(hasil['penyakit_info']['mask_disease'], 
                                       cv2.COLOR_GRAY2RGB)
            st.image(disease_viz, caption="Mask Bercak Penyakit", 
                    use_container_width=True, clamp=True)
        with col_t5b:
            st.image(hasil['result'], caption="Overlay Hasil Akhir", 
                    use_container_width=True, clamp=True)
        st.markdown("**Deskripsi**: Visualisasi akhir dengan marking kontur daun (hijau) dan area bercak (merah)")

# Footer
st.markdown("---")
st.markdown("""
<div class='footer'>
    <h3>🍃 Sistem Deteksi Penyakit Daun Tanaman</h3>
    <p><strong>Teknologi:</strong> Computer Vision & Image Processing dengan Kalibrasi Adaptif</p>
    <p><strong>📚 Metode:</strong> Segmentasi → Enhancement → Morfologi → Analisis Warna HSV + LAB → Analisis Tekstur → Klasifikasi</p>
    <p>Dikembangkan menggunakan OpenCV, NumPy, dan Streamlit</p>
    <p style='margin-top: 15px; color: #666;'>
        💡 <strong>Cara Kerja:</strong><br>
        1. <strong>Kalibrasi warna dasar</strong> - Sistem menyesuaikan threshold berdasarkan warna alami tanaman<br>
        2. <strong>Segmentasi daun</strong> - Memisahkan daun dari background<br>
        3. <strong>Analisis multi-channel</strong> - HSV untuk warna, LAB untuk kecerahan, tekstur untuk pola<br>
        4. <strong>Filter berbasis pola</strong> - Bercak = spot tidak merata (bukan warna seragam)<br>
        5. <strong>Connected components</strong> - Filter noise & false positive<br>
        6. <strong>Klasifikasi adaptif</strong> - Tingkat kesehatan berdasarkan persentase bercak
    </p>
    <p style='margin-top: 10px; padding: 10px; background-color: #fff3cd; border-radius: 5px;'>
        ⚠️ <strong>Solusi untuk False Positive:</strong><br>
        Jika tanaman memiliki warna dasar kuning/merah alami, pilih mode kalibrasi yang sesuai di sidebar.
        Sistem akan fokus mendeteksi <strong>bercak tidak merata</strong> (coklat gelap/hitam) 
        daripada warna kuning seragam yang merupakan karakteristik alami tanaman.
    </p>
</div>
""", unsafe_allow_html=True)