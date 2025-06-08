try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.applications.efficientnet import preprocess_input
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow-cpu"])
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.applications.efficientnet import preprocess_input

# Set page config
st.set_page_config(
    page_title="Deteksi Penyakit Daun Mangga",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load model
model = load_model('model/mangalyze_model.h5')

# Label and recommendation maps
label_map = {
    0: 'Anthracnose',
    1: 'Bacterial Canker',
    2: 'Cutting Weevil',
    3: 'Die Back',
    4: 'Gall Midge',
    5: 'Healthy',
    6: 'Powdery Mildew',
    7: 'Sooty Mould'
}

recommendation_map = {
    'Anthracnose': 'Gunakan fungisida berbahan aktif (mankozeb, tembaga hidroksida, atau propineb) sesuai dosis anjuran.',
    'Bacterial Canker': 'Potong bagian yang terinfeksi dan gunakan bakterisida berbahan tembaga (copper-based).',
    'Cutting Weevil': 'Gunakan insektisida berbahan aktif (imidakloprid, lambda-cyhalothrin) dan periksa kebersihan lingkungan sekitar tanaman.',
    'Die Back': 'Lakukan pemangkasan daun mati dan semprotkan fungisida sistemik (benomil, karbendazim, tebuconazole).',
    'Gall Midge': 'Pangkas dan bakar daun/bunga yang terinfestasi dan aplikasikan insektisida sistematik (imidakloprid, abamektin, spinosad).',
    'Healthy': 'Tanaman sehat! Lanjutkan pemupukan dan penyiraman rutin.',
    'Powdery Mildew': 'Semprot dengan fungisida sistemik dan preventif (karathane, hexaconazole, sulfur, miklobutanil).',
    'Sooty Mould': 'Pangkas ranting yang terlalu rimbun dan semprot air sabun ringan atau campuran air + fungisida ringan.'
}

def extract_features(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# Custom CSS to style the app
st.markdown("""
    <style>
        /* Font */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
        
        * {
            font-family: 'Poppins', sans-serif;
        }
        
        /* Navbar Fix */
        .navbar {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1000;
            background-color: #f8f9fa;
            padding: 10px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .navbar-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }
        
        .navbar-brand {
            font-weight: bold;
            font-size: 1.5rem;
            color: #178011 !important;
            text-decoration: none;
        }
        
        .navbar-brand span {
            color: #09d65f !important;
        }
        
        .nav-links {
            display: flex;
            gap: 20px;
        }
        
        .nav-link {
            color: #495057;
            text-decoration: none;
            font-weight: 500;
            padding: 5px 0;
            position: relative;
            transition: color 0.3s ease;
        }
        
        .nav-link:hover {
            color: #178011 !important;
        }
        
        .nav-link::after {
            content: '';
            position: absolute;
            width: 0;
            height: 2px;
            bottom: 0;
            left: 0;
            background-color: #178011;
            transition: width 0.3s ease;
        }
        
        .nav-link:hover::after {
            width: 100%;
        }
        
        .nav-link.active {
            color: #178011 !important;
            font-weight: bold;
        }
        
        .nav-link.active::after {
            width: 100%;
        }
        
        /* Pastikan konten tidak tertutup navbar */
        .main-container {
            padding-top: 80px !important;
        }
        
        /* Hero section */
        .hero {
            background: linear-gradient(to right, rgba(70, 129, 92, 0.689), rgba(11, 87, 35, 0.667)), 
                        url("https://images.unsplash.com/photo-1724565923616-efc1864bbacf?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D") center/cover no-repeat;
            color: white;
            padding: 150px 0 100px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        
        .hero h1 {
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 20px;
        }
        
        .hero p {
            font-size: 1.2rem;
            margin-bottom: 30px;
        }
        
        .hero-btn {
            background-color: white;
            color: #178011;
            padding: 12px 30px;
            border-radius: 5px;
            text-decoration: none;
            font-weight: 600;
            transition: all 0.3s ease;
            display: inline-block;
        }
        
        .hero-btn:hover {
            background-color: #f8f9fa;
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        /* Disease cards */
        .card-disease {
            height: 100%;
            display: flex;
            flex-direction: column;
            transition: all 0.3s ease;
            cursor: pointer;
            border-radius: 1rem !important;
            box-shadow: 0 0.5rem 1rem rgb(0 0 0 / 0.15);
            padding: 20px;
            margin-bottom: 20px;
            background-color: white;
            border: 1px solid #e9ecef;
        }
        
        .card-disease:hover {
            transform: translateY(-10px);
            box-shadow: 0 1rem 2rem rgb(0 0 0 / 0.25);
        }
        
        .card-disease i {
            font-size: 3rem;
            color: #0f8835;
            margin-bottom: 1rem;
        }
        
        .card-disease h3 {
            color: #212529;
            margin-bottom: 15px;
        }
        
        .card-disease p {
            color: #6c757d;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #178011;
            color: white;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease;
            width: 100%;
            padding: 12px;
        }
        
        .stButton>button:hover {
            background-color: #0c6c2c;
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        /* Result styling */
        .result-image {
            max-height: 300px;
            border-radius: 1rem;
            box-shadow: 0 0.25rem 0.5rem rgb(0 0 0 / 0.15);
            margin: 0 auto;
            display: block;
        }
        
        .recommendation-card {
            border-radius: 1rem;
            box-shadow: 0 0.25rem 0.75rem rgb(0 0 0 / 0.1);
            padding: 20px;
            background-color: #f8f9fa;
            margin-top: 20px;
        }
        
        /* Section titles */
        .section-title {
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 30px;
            color: #212529;
        }
        
        .section-subtitle {
            text-align: center;
            color: #6c757d;
            margin-bottom: 30px;
        }
        
        /* Hide streamlit default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* File uploader styling */
        .stFileUploader>div>div>div>div {
            color: #495057;
        }
        
        .uploaded-image {
            border-radius: 10px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Add Bootstrap Icons
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css" rel="stylesheet">
""", unsafe_allow_html=True)

# Navbar with fixed scrolling behavior
st.markdown("""
    <nav class="navbar">
        <div class="navbar-container">
            <a class="navbar-brand" href="#hero">Mangalyze<span>.</span></a>
            <div class="nav-links">
                <a class="nav-link" href="#hero">Beranda</a>
                <a class="nav-link" href="#disease-cards">Jenis Penyakit</a>
                <a class="nav-link" href="#deteksi">Prediksi</a>
            </div>
        </div>
    </nav>
""", unsafe_allow_html=True)

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Hero Section
st.markdown("""
    <section id="hero" class="hero">
        <h1>Deteksi Penyakit Daun Mangga</h1>
        <p>Gunakan AI untuk mengidentifikasi penyakit pada daun mangga dengan cepat dan akurat.</p>
        <a href="#deteksi" class="hero-btn">Mulai Deteksi</a>
    </section>
""", unsafe_allow_html=True)

# Diseases Section
st.markdown("""
    <section id="disease-cards">
        <h2 class="section-title">Jenis Penyakit Daun Mangga</h2>
        <p class="section-subtitle">Berikut beberapa penyakit yang bisa dideteksi sistem kami:</p>
""", unsafe_allow_html=True)

# Disease cards in columns
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
        <div class="card-disease">
            <i class="bi bi-bug-fill"></i>
            <h3>Anthracnose</h3>
            <p>Penyakit jamur (<b>Colletotrichum gloeosporioides</b>) yang ditandai dengan munculnya bercak-bercak coklat tua hingga hitam.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="card-disease">
            <i class="bi bi-tree-fill"></i>
            <h3>Die Back</h3>
            <p>Penyakit jamur yang menyebabkan kematian jaringan tanaman dimulai dari ujung (pucuk) ranting atau cabang.</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="card-disease">
            <i class="bi bi-droplet-fill"></i>
            <h3>Bacterial Canker</h3>
            <p>Penyakit berbasis bakteri yang ditandai dengan munculnya bercak nekrotik (mati) berair atau kering pada daun.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="card-disease">
            <i class="bi bi-bug"></i>
            <h3>Gall Midge</h3>
            <p>Serangan hama yang menyebabkan terbentuknya "gall" (bengkokan atau benjolan abnormal) pada daun.</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="card-disease">
            <i class="bi bi-cloud-fog2-fill"></i>
            <h3>Powdery Mildew</h3>
            <p>Penyakit jamur (<b>Oidium mangiferae</b>) yang ditandai dengan munculnya lapisan putih seperti tepung pada permukaan daun.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="card-disease">
            <i class="bi bi-heart-fill" style="color: #0f8835;"></i>
            <h3>Healthy</h3>
            <p>Kondisi daun mangga yang sehat tanpa tanda-tanda penyakit atau serangan hama.</p>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
        <div class="card-disease">
            <i class="bi bi-bug-fill"></i>
            <h3>Cutting Weevil</h3>
            <p>Serangan hama dari kelompok kumbang kecil yang menyerang tunas muda, tangkai bunga, dan cabang kecil.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="card-disease">
            <i class="bi bi-journal-medical"></i>
            <h3>Sooty Mould</h3>
            <p>Penyakit jamur yang menyebabkan lapisan berwarna hitam pekat seperti arang atau jelaga muncul di permukaan daun.</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("</section>", unsafe_allow_html=True)

# Detection Section
st.markdown("""
    <section id="deteksi">
        <h2 class="section-title">Mulai Prediksi Penyakit</h2>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([5, 7])

with col_left:
    st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0.5rem 1rem rgb(0 0 0 / 0.15);">
            <h3 style="margin-bottom: 20px;">Unggah Gambar Daun Mangga</h3>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=False, key="file_uploader", label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True, output_format="auto", clamp=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0.5rem 1rem rgb(0 0 0 / 0.15); height: 100%;">
            <h3 style="text-align: center; color: #178011; margin-bottom: 20px;">
                <i class="bi bi-clipboard-check" style="margin-right: 10px;"></i>Hasil Deteksi
            </h3>
    """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        if st.button("Deteksi Sekarang"):
            with st.spinner("Menganalisis gambar..."):
                # Save the uploaded file
                filename = f"{int(time.time())}_{uploaded_file.name}"
                image_path = os.path.join("temp", filename)
                os.makedirs("temp", exist_ok=True)
                
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process the image
                image = extract_features(image_path)
                prediction = model.predict(image)
                predicted_label = np.argmax(prediction)
                label_name = label_map[predicted_label]
                confidence = prediction[0][predicted_label] * 100
                final_result = f"{label_name} ({confidence:.2f}%)"
                recommendation = recommendation_map.get(label_name, "Tidak ada rekomendasi khusus.")
                
                # Display results
                st.image(image_path, caption="Gambar daun mangga yang dideteksi", width=300, use_column_width=True)
                
                # Diagnosis
                st.markdown(f"""
                    <div style="display: flex; align-items: center; background-color: #e8f5e9; padding: 15px; border-radius: 10px; margin-top: 20px;">
                        <i class="bi bi-activity" style="font-size: 2rem; margin-right: 15px; color: #178011;"></i>
                        <div>
                            <h4 style="margin-bottom: 5px;">Diagnosa</h4>
                            <p style="font-size: 1.2rem; font-weight: 600;">{final_result}</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Recommendation
                st.markdown(f"""
                    <div class="recommendation-card">
                        <h4 style="color: #178011; margin-bottom: 15px;">
                            <i class="bi bi-lightbulb-fill" style="margin-right: 10px;"></i>Rekomendasi Penanganan
                        </h4>
                        <p>{recommendation}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Remove temporary file
                os.remove(image_path)
    else:
        st.markdown("""
            <p style="text-align: center; color: #6c757d; font-style: italic; margin-top: 50px;">
                Unggah gambar dan klik "Deteksi Sekarang" untuk memulai.
            </p>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</section>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: #f8f9fa; margin-top: 50px; border-radius: 10px;">
        <p>&copy; 2025 Mangalyze. Dibuat oleh Tim Mangalyze.</p>
    </div>
    </div> <!-- Close main-container -->
""", unsafe_allow_html=True)

# Fixed JavaScript for smooth scrolling and active state
st.markdown("""
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Fungsi untuk handle smooth scroll
            function smoothScroll(target) {
                const element = document.querySelector(target);
                if (element) {
                    window.scrollTo({
                        top: element.offsetTop - 70,
                        behavior: 'smooth'
                    });
                }
            }
            
            // Handle klik nav link
            document.querySelectorAll('.nav-link').forEach(link => {
                link.addEventListener('click', function(e) {
                    e.preventDefault();
                    const target = this.getAttribute('href');
                    smoothScroll(target);
                });
            });
            
            // Handle scroll untuk active state
            window.addEventListener('scroll', function() {
                const scrollPos = window.scrollY + 100;
                document.querySelectorAll('.nav-link').forEach(link => {
                    const target = document.querySelector(link.getAttribute('href'));
                    if (target) {
                        if (target.offsetTop <= scrollPos && 
                            target.offsetTop + target.offsetHeight > scrollPos) {
                            link.classList.add('active');
                        } else {
                            link.classList.remove('active');
                        }
                    }
                });
            });
        });
    </script>
""", unsafe_allow_html=True)
