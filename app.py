from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    jsonify,
)
from db import db, UserModel, HistoryModel
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
from werkzeug.utils import secure_filename
import tensorflow as tf
import os
from PIL import Image
import numpy as np

# App
app = Flask(__name__)
app.secret_key = "buat_secret_key_lebih_rumit"

# Load model
model = tf.keras.models.load_model("MobileNet_TrainTestVal_ACC91%.h5")

# Mendefinisikan classes dan deskripsinya
classes = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
class_descriptions = {
    "cataract": """<span class="fs-6 fw-bold my-2">Description :</span><br>
            <span class="justify-text">Cataract merupakan proses terjadinya opasitas secara progesif pada lensa atau kapsul lensa akibat dari proses penuaan dan dapat timbul juga pada saat kelahiran (katarak kongenital). Selain itu katarak juga dapat timbul karena penggunaan kortikosteroid jangka panjang, penyakit sistematik, pemajanan radiasi, atau kelainan mata seperti uveitis anterior.</span><br>
            <span class="fs-6 fw-bold my-2">ciri-ciri mata cataract:</span><br>
            <ul class="mb-1">
                <li>Terbentuknya lukisan pada mata</li>
                <li>Adanya lapisan kuning atau coklat pada mata</li>
                <li>Mata terasa sakit jika terkena cahaya yang menyilaukan</li>
                <li>Pandangan mata kabur</li>
                <li>Gambar menjadi ganda</li>
            </ul>
            <span class="fs-5 fw-bold my-2 mb-2">Penanganan penyakit cataract</span><br>
            <span class="fs-6 fw-bold my-3">1. Pembedahan sebagai Penanganan Utama</span><br>
            <span class="justify-text">Pembedahan dengan teknik fakoemulsifikasi dan implantasi lensa intraokular adalah metode yang terbukti efektif untuk mengatasi katarak. Pembedahan katarak dapat meningkatkan kualitas hidup dan mengurangi risiko jatuh pada pasien dengan degenerasi makula terkait usia (AMD) </span><br>
            <span class="fs-6 fw-bold my-3">2. Pengobatan Medis:</span><br>
            <span class="justify-text">Beberapa obat anti-katarak seperti carnosine telah mencapai uji klinis dan menunjukkan hasil yang menjanjikan, meskipun masih memerlukan penelitian lebih lanjut. <br>
            Pengobatan farmakologis untuk pencegahan katarak terhambat oleh banyak hambatan fisiologis yang harus diatasi oleh agen terapeutik untuk mencapai lensa avaskular</span>
        """,
    "diabetic_retinopathy": """<span class="fs-6 fw-bold my-2">Description :</span><br>
            <span class="justify-text">Diabetic retinopathy merupakan kerusakan pada retina mata yang disebabkan oleh komplikasi dari penyakit diabetes melitus. Komplikasi ini menyebabkan penyumbatan ada pembuluh darah di bagian retina mata.</span><br>
            <span class="fs-6 fw-bold my-2">ciri-ciri mata diabetic retinopathy:</span><br>
            <ul class="mb-1">
                <li>Pembengkakan dan tumpukan darah atau lemak di mata</li>
                <li>Terlepasnya retina (ablasi retina)</li>
                <li>Kelainan di saraf mata</li>
                <li>Pembuluh darah yang tidak normal</li>
                <li>Perdarahan di bagian tengah bola mata</li>
            </ul>
            <span class="fs-5 fw-bold my-2 mb-2">Penanganan penyakit diabetic retinopathy</span><br>
            <span class="fs-6 fw-bold my-3">1. Terapi Farmakologis</span><br>
            <ul class="mb-1">
                <li>Agen Anti-VEGF: Injeksi intravitreal agen anti-VEGF efektif dalam mengobati edema makula diabetik (DME) dan retinopati diabetik proliferatif (PDR).</li>
                <li>Kortikosteroid: Tetes mata kortikosteroid dan injeksi intravitreal telah menunjukkan efek menguntungkan dalam mengelola DME.</li>
                <li>Agen Sistemik: Kontrol glikemik yang intensif, pengendalian dislipidemia, dan antagonis sistem renin-angiotensin bermanfaat untuk retinopati diabetik (DR) dan DME.</li>
            </ul>
            <span class="fs-6 fw-bold my-2">2. Terapi Laser:</span><br>
            <ul class="mb-1">
                <li>Fotokoagulasi Panretinal (PRP): Efektif dalam mengurangi risiko kehilangan penglihatan berat pada PDR dan direkomendasikan untuk tahap proliferatif berisiko tinggi.</li>
                <li>Fotokoagulasi Laser Fokal/Grid: Efektif dalam mengurangi risiko kehilangan penglihatan sedang pada mata dengan edema makula.</li>
            </ul>
        """,
    "glaucoma": """ <span class="fs-6 fw-bold my-2">Description :</span><br>
            <span class="justify-text">Glaucoma merupakan peradangan pada optik mata yang ditandai dengan kemunduran progesif dari kepala saraf optik dan luas pandangan.</span><br>
            <span class="fs-6 fw-bold my-2">ciri-ciri mata glaucoma:</span><br>
            <ul class="mb-1">
                <li>Terdapat lingkaran seperti pelangi ketika melihat ke arah cahaya terang</li>
                <li>Sakit kepala berat </li>
                <li>Suka muntah tiba-tiba</li>
                <li>Kornea mata tidak jernih</li>
                <li>Suka muncul bintik-bintik hitam pada pandangan</li>
            </ul>
            <span class="fs-5 fw-bold my-2 mb-2">Penanganan penyakit glaucoma</span><br>
            <span class="fs-6 fw-bold my-3">1. Terapi Medis</span><br>
            <ul class="mb-1">
                <li>Obat-obatan topikal seperti beta-blocker, prostaglandin analog, alpha-agonists, dan inhibitor karbonat anhidrase adalah pilihan pertama dalam terapi glaukoma</li>
                <li>Obat-obatan baru seperti netarsudil dan latanoprostene bunod telah disetujui dan menawarkan mekanisme baru untuk menurunkan IOP</li>
            </ul>
            <span class="fs-6 fw-bold my-2">2. Terapi Laser:</span><br>
            <ul class="mb-1">
                <li>Trabekuloplasti laser selektif (SLT) telah terbukti efektif sebagai terapi lini pertama untuk glaukoma sudut terbuka dan hipertensi okular, dengan hasil yang lebih baik dalam kontrol IOP dibandingkan dengan obat tetes mata</li>
                <li>Cyclophotocoagulation laser adalah metode lain yang digunakan untuk menurunkan produksi humor aqueous dan menurunkan IOP, meskipun bukti efektivitasnya masih terbatas</li>
            </ul>
        """,
    "normal": """<span class="fs-6 fw-bold my-2">Description :</span><br>
            <span class="justify-text">Mata normal merupakan kondisi dimana mata berfungsi dengan baik tanpa adanya kelainan atau penyakit. Penglihatan jelas, tajam, dan tidak ada gangguan seperti kabur atau distorsi.</span>
        """,
}


# Tentukan ekstensi file yang diizinkan
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


# Berfungsi untuk memeriksa apakah ekstensi file diperbolehkan
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Konfigurasi SQLAlchemy
app.config["SQLALCHEMY_DATABASE_URI"] = (
    "mysql://root:@localhost/flask_deteksipenyakitmata"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)  # Inisialisasi SQLAlchemy dengan aplikasi Flask

# Buat direktori untuk menyimpan file upload jika belum ada
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# menampilkan data
@app.route("/")
def home():
    return render_template("user/index.html")


# menampilkan halaman admin
@app.route("/admin/image-clasification")
def admin():
    if "email" not in session:
        return redirect(url_for("login"))
    else:
        return render_template("admin/imageClasification.html")


@app.route("/register", methods=["POST", "GET"])
def register():
    if request.method == "GET":
        return render_template("auth/register.html", title="Register")
    else:
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        # Periksa apakah semua kolom terisi
        if not name or not email or not password:
            error = "Silakan isi semua kolom."
            return render_template("auth/register.html", error=error)

        # Cek email jika sudah ada didatabase
        existing_user = UserModel.query.filter_by(email=email).first()
        if existing_user:
            error = "Email telah terdaftar. Silakan gunakan email lain."
            return render_template("auth/register.html", error=error)

        # Enkripsi kata sandi sebelum menyimpannya
        hashed_password = generate_password_hash(password)

        # buat user baru
        new_user = UserModel(name=name, email=email, password=hashed_password)

        # Simpan
        db.session.add(new_user)
        db.session.commit()

        # Set session setelah register
        session["name"] = name
        session["email"] = email
        session["user_id"] = new_user.id

        return redirect(url_for("admin"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == ("POST"):
        email = request.form["email"]
        password = request.form["password"]

        # Mencari pengguna berdasarkan alamat email
        user = db.session.query(UserModel).filter_by(email=email).first()

        if user:
            # Memeriksa kata sandi
            if check_password_hash(user.password, password):
                # Jika kata sandi cocok, atur sesi dan arahkan ke halaman yang sesuai
                session["name"] = user.name
                session["email"] = user.email
                session["user_id"] = user.id
                return redirect(url_for("admin"))
            else:
                # Jika kata sandi tidak cocok, beri pesan kesalahan
                flash("Gagal, Email dan Password Tidak Cocok")
                return redirect(url_for("login"))
        else:
            # Jika pengguna tidak ditemukan, beri pesan kesalahan
            flash("Gagal, Email tidak terdaftar")
            return redirect(url_for("login"))
    else:
        return render_template("auth/login.html", title="Login")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# Route untuk menangani image upload dan classification
@app.route("/predict", methods=["POST"])
def predict():
    if "email" not in session:
        return redirect(url_for("login"))
    else:
        if "file" not in request.files:
            return jsonify({"error": "No file part"})

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No selected file"})

        if file and allowed_file(file.filename):
            # Generate filename unique
            original_filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{original_filename}"
            filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
            file.save(filepath)

            try:
                # Preprocess the image
                img = Image.open(filepath).convert("RGB")
                img = img.resize((256, 256))
                img = np.array(img) / 255.0
                img = np.expand_dims(img, axis=0)

                # Perform classification
                predictions = model.predict(img)
                predicted_class = np.argmax(predictions)
                predicted_label = classes[predicted_class]
                accuracy = predictions[0][predicted_class] * 100

                # Get description for predicted label
                description = class_descriptions[predicted_label]

                # Save prediction to the database
                user_id = session.get("user_id")
                if user_id is None:
                    return redirect(url_for("login"))

                new_history = HistoryModel(
                    user_id=user_id,
                    file_name=unique_filename,
                    result=predicted_label,
                    score=f"{accuracy:.2f}%",
                )
                db.session.add(new_history)
                db.session.commit()

                return jsonify(
                    {
                        "predicted_label": predicted_label,
                        "description": description,
                        "accuracy": f"{accuracy:.2f}%",
                    }
                )

            except Exception as e:
                return jsonify({"error": str(e)})

        else:
            return jsonify({"error": "File type not allowed"})


# history
@app.route("/admin/history", methods=["GET"])
def history():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]
    user_history = db.session.query(HistoryModel).filter_by(user_id=user_id).all()
    return render_template("admin/history.html", history=user_history)


# delete history
@app.route("/deleteHistory/<int:id>")
def deleteHistory(id):
    user_id = session["user_id"]
    history = (
        db.session.query(HistoryModel)
        .filter_by(user_id=user_id)
        .filter_by(id=id)
        .first()
    )

    if history:
        # Path file
        file_path = os.path.join(UPLOAD_FOLDER, history.file_name)

        # hapus file jika ada
        if os.path.exists(file_path):
            os.remove(file_path)

        # hapus yang ada didatabase
        db.session.delete(history)
        db.session.commit()

        # Set flash message
        flash("History berhasil dihapus.", "success")

    return redirect("/admin/history")


if __name__ == "__main__":
    app.run(debug=True)
