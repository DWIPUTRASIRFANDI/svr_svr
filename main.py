import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn import metrics
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Prediksi Penjualan Barang Almey Petshop", layout="wide")
# Create menu
selected = option_menu(
    menu_title=None,
    options=["Home", "Data Visualisation", "Prediction"],
    icons=["house", "book", "calculator"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

#row0_spacer1, row0_1, row0_spacer2= st.columns((0.1, 3.2, 0.1))
#row1_spacer1, row1_1, row1_spacer2, row1_2 = st.columns((0.1, 1.5, 0.1, 1.5))
#row1_spacer1, row1_1, row1_spacer2 = st.columns((0.1, 3.2, 0.1))
#row0_spacer3, row3_0, row0_spacer3= st.columns((0.1, 3.2, 0.1))

row0_spacer1, row0_1, row0_spacer2 = st.columns((0.1, 3.2, 0.1))
row1_spacer1, row1_1, row1_spacer2, row1_2 = st.columns((0.1, 1.5, 0.1, 1.5))
row0_spacer3, row3_0, row0_spacer4 = st.columns((0.1, 3.2, 0.1))

# Load dataset
df = pd.read_csv('dataset_new1.csv')

# Model
model = pd.read_pickle('model_svr_new.pkl')

# Handle selected option
if selected == "Home":
    row0_1.title("Aplikasi Prediksi Penjualan Barang Almey Petshop menggunakan Support Vector Regression (SVR)")
    with row0_1:
        st.markdown(
            "Aplikasi Prediksi Penjualan Barang Almey Petshop menggunakan Support Vector Regression adalah sebuah sistem yang dirancang untuk membantu Almey Petshop dalam memprediksi penjualan barang mereka di masa depan. Metode yang digunakan adalah Support Vector Regression (SVR), sebuah teknik dalam machine learning yang dapat digunakan untuk membangun model prediksi berdasarkan pola-pola data historis. Berikut adalah deskripsi umum tentang bagaimana aplikasi ini bekerja:"
        )
        st.write('**Berikut adalah deskripsi umum tentang bagaimana aplikasi ini bekerja:**')
        st.markdown("1. **Input Data**: Aplikasi akan membutuhkan data historis penjualan barang-barang Almey Petshop. Data ini akan mencakup berbagai variabel, seperti tanggal, jenis barang, harga, cuaca, promosi, dan faktor-faktor lain yang mungkin memengaruhi penjualan.")
        st.markdown("2. **Preprocessing**: Sebelum membangun model, data akan diproses untuk membersihkan data yang tidak lengkap atau tidak relevan. Ini mungkin melibatkan langkah-langkah seperti penghapusan data duplikat, penanganan nilai-nilai yang hilang, dan normalisasi data jika diperlukan.")
        st.markdown("3. **Feature Selection**: Setelah preprocessing, aplikasi akan memilih fitur-fitur yang paling relevan untuk digunakan dalam memprediksi penjualan. Ini dapat dilakukan dengan menggunakan teknik analisis statistik atau pemilihan fitur berbasis domain knowledge.")
        st.markdown("4. **Model Building**: Dengan menggunakan algoritma Support Vector Regression (SVR), aplikasi akan membangun model prediksi berdasarkan data latih yang telah diproses. SVR bekerja dengan mencari garis atau permukaan terbaik yang memisahkan titik-titik data dalam dimensi yang tinggi.")
        st.markdown("5. **Validasi Model**: Model yang dibangun akan divalidasi menggunakan data yang tidak terlihat sebelumnya untuk memastikan kinerjanya yang baik dan menghindari overfitting.")
        st.markdown("6. **Prediksi Penjualan**: Setelah model divalidasi, aplikasi akan siap untuk digunakan dalam memprediksi penjualan barang-barang Almey Petshop di masa depan. Input yang diberikan mungkin termasuk tanggal tertentu, kondisi cuaca, promosi yang sedang berjalan, dan faktor-faktor lain yang relevan.")
        st.markdown("7. **Evaluasi dan Pemantauan**: Performa model akan terus dipantau dan dievaluasi secara berkala. Jika diperlukan, model dapat disesuaikan atau diperbarui dengan data baru untuk meningkatkan akurasinya seiring waktu.")
        st.write('')
        st.write('**Dataset:**')
        st.write(df.head())

elif selected == "Data Visualisation":
    # Data Visualisasi dengan plotly
    with row1_1:
        st.subheader('Pilih fitur yang ingin ditampilkan histogramnya')
        fitur = st.selectbox('Fitur', ('Stok_1', 'Stok_2', 'Stok_3', 'Stok_4', 'Stok_5', 'Stok_6', 'Stok_7', 'Stok_8', 'Stok_9', 'Stok_10', 'Stok_11', 'Stok_12', 'Stok_13', 'Stok_14', 'Stok_15'))
        fig = px.histogram(df, x=fitur, marginal='box', hover_data=df.columns)
        st.plotly_chart(fig)
    with row1_2:
        st.subheader('Pilih fitur yang ingin ditampilkan scatter plotnya')
        fitur1 = st.selectbox('Fitur 1', ('Stok_1', 'Stok_2', 'Stok_3', 'Stok_4', 'Stok_5', 'Stok_6', 'Stok_7', 'Stok_8', 'Stok_9', 'Stok_10', 'Stok_11', 'Stok_12', 'Stok_13', 'Stok_14', 'Stok_15'))
        fitur2 = st.selectbox('Fitur 2', ('Stok_1', 'Stok_2', 'Stok_3', 'Stok_4', 'Stok_5', 'Stok_6', 'Stok_7', 'Stok_8', 'Stok_9', 'Stok_10', 'Stok_11', 'Stok_12', 'Stok_13', 'Stok_14', 'Stok_15'))
        fig = px.scatter(df, x=fitur1, y=fitur2, color='Stok_15', hover_data=df.columns)
        st.plotly_chart(fig)

elif selected == "Prediction":
    with row0_1:
        st.subheader('Pengaturan Variabel')
    with row1_1:
        option = st.selectbox("Pilih Variabel Dependent", ('', 'Stok_15'))
    with row3_0:
        button = st.button('Predict')
        if button:
            X = df.drop(['Kode_Barang', 'Nama_Barang'], axis=1)
            X = X.drop([option], axis=1)
            y = df[option]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            model = joblib.load('model_svr_new.pkl')
            y_pred = model.predict(X_test)
            
            st.write('**Hasil Prediksi Penjualan pada bulan mendatang**')
            result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            st.write(result)

            st.write('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_pred),4))
            st.write('Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred),4))
            st.write('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),4))
            st.write('Coefficient of determination:', round(metrics.r2_score(y_test, y_pred),4))
            st.write('')

            st.markdown("Analisis hasil metrik yang diberikan memberikan gambaran tentang seberapa baik model prediksi penjualan barang Almey Petshop menggunakan Support Vector Regression (SVR) dalam memprediksi penjualan. Berikut adalah penjelasan untuk setiap metrik:")
            st.markdown("1. **Mean Absolute Error (MAE):** MAE adalah rata-rata dari selisih absolut antara nilai prediksi dan nilai sebenarnya. Nilai MAE yang rendah menunjukkan bahwa prediksi model cukup akurat. Dalam hal ini, MAE sekitar 0.066 menunjukkan bahwa rata-rata kesalahan prediksi model adalah sekitar 0.066 unit, yang cukup kecil.")
            st.markdown("2. **Mean Squared Error (MSE):** MSE adalah rata-rata dari selisih kuadrat antara nilai prediksi dan nilai sebenarnya. Nilai MSE yang rendah menunjukkan bahwa model memiliki sedikit kesalahan dan penalti yang lebih besar diberikan pada kesalahan yang lebih besar. MSE sekitar 0.0056 menunjukkan bahwa kesalahan kuadrat rata-rata model juga sangat rendah.")
            st.markdown("3. **Root Mean Squared Error (RMSE):** RMSE adalah akar dari MSE dan memberikan ukuran yang lebih mudah diinterpretasikan karena berada dalam skala yang sama dengan data asli. RMSE sekitar 0.075 menunjukkan bahwa kesalahan prediksi model secara rata-rata adalah sekitar 0.075 unit, yang menunjukkan akurasi model yang baik.")
            st.markdown("4. **Coefficient of Determination (R-squared):** R² mengukur seberapa baik model menjelaskan variasi dalam data yang diamati. Nilai R² berkisar antara 0 dan 1, dengan 1 menunjukkan bahwa model menjelaskan seluruh variasi dalam data. Nilai R² sebesar 1.00 menunjukkan bahwa model Anda mampu menjelaskan 100% variasi dalam data yang diamati, yang merupakan hasil yang sangat ideal dan menunjukkan model yang sempurna.")
            st.markdown("Secara keseluruhan, metrik-metrik ini menunjukkan bahwa model yang Anda gunakan memiliki kinerja yang sangat baik dengan kesalahan prediksi yang sangat rendah dan kemampuan yang sempurna dalam menjelaskan variasi data. Namun, penting untuk memastikan bahwa model ini tidak overfitting dengan memverifikasi kinerjanya pada data uji atau validasi yang independen.")
