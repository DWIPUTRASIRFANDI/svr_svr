import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn import metrics
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Prediksi Penjualan Tanaman", layout="wide")
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
df1 = pd.read_csv('new_dataset_penjualan.csv')
df2 = pd.read_csv('new_dataset_stock.csv')

# Model
model1 = pd.read_pickle('model_svr_penjualan.pkl')
model2 = pd.read_pickle('model_svr_stock.pkl')

# Handle selected option
if selected == "Home":
    row0_1.title("Aplikasi Prediksi Penjualan Tanaman menggunakan Support Vector Regression (SVR)")
    with row0_1:
        st.markdown(
            "Aplikasi Prediksi Penjualan Tanaman adalah sebuah sistem berbasis web yang dirancang untuk membantu petani, pengecer tanaman, dan pelaku bisnis agrikultur dalam memprediksi penjualan tanaman berdasarkan data historis. Sistem ini memanfaatkan algoritma Support Vector Regression (SVR), yang dikenal efektif dalam menangani data non-linear dan menghasilkan prediksi yang akurat."
        )
        st.write('**Berikut adalah deskripsi umum tentang bagaimana aplikasi ini bekerja:**')
        st.markdown("1. **Input Data**: Aplikasi akan membutuhkan data historis penjualan tanaman yang mungkin memengaruhi penjualan.")
        st.markdown("2. **Preprocessing**: Sebelum membangun model, data akan diproses untuk membersihkan data yang tidak lengkap atau tidak relevan. Ini mungkin melibatkan langkah-langkah seperti penghapusan data duplikat, penanganan nilai-nilai yang hilang, dan normalisasi data jika diperlukan.")
        st.markdown("3. **Feature Selection**: Setelah preprocessing, aplikasi akan memilih fitur-fitur yang paling relevan untuk digunakan dalam memprediksi penjualan. Ini dapat dilakukan dengan menggunakan teknik analisis statistik atau pemilihan fitur berbasis domain knowledge.")
        st.markdown("4. **Model Building**: Dengan menggunakan algoritma Support Vector Regression (SVR), aplikasi akan membangun model prediksi berdasarkan data latih yang telah diproses. SVR bekerja dengan mencari garis atau permukaan terbaik yang memisahkan titik-titik data dalam dimensi yang tinggi.")
        st.markdown("5. **Validasi Model**: Model yang dibangun akan divalidasi menggunakan data yang tidak terlihat sebelumnya untuk memastikan kinerjanya yang baik dan menghindari overfitting.")
        st.markdown("6. **Prediksi Penjualan**: Setelah model divalidasi, aplikasi akan siap untuk digunakan dalam memprediksi penjualan tanaman di masa depan. Input yang diberikan mungkin termasuk tanggal tertentu, kondisi cuaca, promosi yang sedang berjalan, dan faktor-faktor lain yang relevan.")
        st.markdown("7. **Evaluasi dan Pemantauan**: Performa model akan terus dipantau dan dievaluasi secara berkala. Jika diperlukan, model dapat disesuaikan atau diperbarui dengan data baru untuk meningkatkan akurasinya seiring waktu.")
        st.write('')
        st.write('**Dataset Penjualan:**')
        st.write(df1.head())
        st.write('')
        st.write('**Dataset Stok:**')
        st.write(df2.head())

elif selected == "Data Visualisation":
    # Data Visualisasi dengan plotly
    row0_1.title("Visualisasi Data Penjualan & Stok")
    with row1_1:
        st.subheader('Histogram Penjualan')
        fitur = st.selectbox('Fitur', ('penjualan_1', 'penjualan_2', 'penjualan_3', 'penjualan_4', 'penjualan_5', 'penjualan_6', 'penjualan_7', 'penjualan_8', 'penjualan_9', 'penjualan_10', 'penjualan_11', 'penjualan_12'))
        fig = px.histogram(df1, x=fitur, marginal='box', hover_data=df1.columns)
        st.plotly_chart(fig)

        st.write('')
        st.write('')

        st.subheader('Histogram Stok')
        fitur = st.selectbox('Fitur', ('stock_1', 'stock_2', 'stock_3', 'stock_4', 'stock_5', 'stock_6', 'stock_7', 'stock_8', 'stock_9', 'stock_10', 'stock_11', 'stock_12'))
        fig = px.histogram(df2, x=fitur, marginal='box', hover_data=df2.columns)
        st.plotly_chart(fig)
    with row1_2:
        st.subheader('Scatter Plot Penjualan')
        fitur1 = st.selectbox('Fitur 1', ('penjualan_1', 'penjualan_2', 'penjualan_3', 'penjualan_4', 'penjualan_5', 'penjualan_6', 'penjualan_7', 'penjualan_8', 'penjualan_9', 'penjualan_10', 'penjualan_11', 'penjualan_12'))
        fitur2 = st.selectbox('Fitur 2', ('penjualan_1', 'penjualan_2', 'penjualan_3', 'penjualan_4', 'penjualan_5', 'penjualan_6', 'penjualan_7', 'penjualan_8', 'penjualan_9', 'penjualan_10', 'penjualan_11', 'penjualan_12'))
        fig = px.scatter(df1, x=fitur1, y=fitur2, color='penjualan_12', hover_data=df1.columns)
        st.plotly_chart(fig)

        st.write('')
        st.write('')

        st.subheader('Scatter Plot Stok')
        fitur1 = st.selectbox('Fitur 1', ('stock_1', 'stock_2', 'stock_3', 'stock_4', 'stock_5', 'stock_6', 'stock_7', 'stock_8', 'stock_9', 'stock_10', 'stock_11', 'stock_12'))
        fitur2 = st.selectbox('Fitur 2', ('stock_1', 'stock_2', 'stock_3', 'stock_4', 'stock_5', 'stock_6', 'stock_7', 'stock_8', 'stock_9', 'stock_10', 'stock_11', 'stock_12'))
        fig = px.scatter(df2, x=fitur1, y=fitur2, color='stock_12', hover_data=df2.columns)
        st.plotly_chart(fig)
        

elif selected == "Prediction":
    with row0_1:
        st.subheader('Pengaturan Variabel')
    with row1_1:
        option = st.selectbox("Pilih Variabel Dependent Penjualan", ('', 'penjualan_12'))
    with row1_2:
        option1 = st.selectbox("Pilih Variabel Dependent Stok", ('', 'stock_12'))
    with row3_0:
        button = st.button('Predict')
        if button and option:
            X = df1.drop(['harga_satuan'], axis=1)
            X = X.drop([option], axis=1)
            y = df1[option]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            
            test = X_test.drop(['nama_tanaman'], axis=1)

            model = joblib.load('model_svr_penjualan.pkl')
            y_pred = model.predict(test)
            
            st.write('**Hasil Prediksi Penjualan pada bulan mendatang**')
            result = pd.DataFrame({'Nama Tanaman':X_test['nama_tanaman'], 'Actual': y_test, 'Predicted': y_pred})
            st.table(result)

            st.write('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_pred),4))
            st.write('Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred),4))
            st.write('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),4))
            st.write('Coefficient of determination:', round(metrics.r2_score(y_test, y_pred),4))
            st.write('')

            st.markdown("Analisis hasil metrik yang diberikan memberikan gambaran tentang seberapa baik model prediksi penjualan Tanaman menggunakan Support Vector Regression (SVR) dalam memprediksi penjualan. Berikut adalah penjelasan untuk setiap metrik:")
            st.markdown("1. **Mean Absolute Error (MAE):** MAE mengukur rata-rata absolut dari selisih antara nilai yang diprediksi dan nilai aktual. Nilai MAE sebesar 1.067 menunjukkan bahwa rata-rata kesalahan prediksi adalah sekitar 1.067 Nilai ini menunjukkan bahwa prediksi model menyimpang sekitar 1.067 unit dari nilai aktualnya. Meskipun kesalahan ini tidak sangat besar, MAE biasanya lebih rendah pada model yang berkinerja baik. Jadi, hasil ini bisa menunjukkan bahwa model masih memiliki ruang untuk perbaikan.")
            st.markdown("2. **Mean Squared Error (MSE):** MSE mengukur rata-rata dari kuadrat selisih antara nilai yang diprediksi dan nilai aktual. MSE memberikan penalti yang lebih besar untuk kesalahan yang lebih besar. Nilai MSE sebesar 1.1442 menunjukkan adanya kesalahan yang relatif besar pada beberapa prediksi. Karena MSE menghukum kesalahan besar lebih berat, ini menunjukkan bahwa model mungkin tidak menangani outlier dengan baik atau memiliki beberapa prediksi yang sangat meleset.")
            st.markdown("3. **Root Mean Squared Error (RMSE):** RMSE adalah akar kuadrat dari MSE dan kembali ke satuan asli dari variabel yang diprediksi. Nilai RMSE sebesar 1.0697 menunjukkan bahwa kesalahan prediksi rata-rata model adalah sekitar 1.07 unit. Ini hampir sama dengan MAE, yang berarti distribusi kesalahan tidak memiliki outlier yang ekstrem tetapi tetap cukup signifikan.")
            st.markdown("4. **Coefficient of Determination (R-squared):** R² biasanya mengukur proporsi variabilitas dalam data yang bisa dijelaskan oleh model, dengan nilai berkisar antara 0 dan 1. Namun, dalam kasus ini, nilai R² adalah negatif. Nilai R² negatif menunjukkan bahwa model tersebut bahkan lebih buruk daripada model baseline (model yang hanya memprediksi rata-rata dari data). Ini menunjukkan bahwa model SVR tidak hanya gagal menangkap pola dalam data, tetapi juga memberikan prediksi yang jauh dari kenyataan.")
            st.markdown("Metrik-metrik ini menunjukkan bahwa model Support Vector Regression (SVR) yang digunakan tidak memberikan hasil yang memuaskan. Nilai MAE dan RMSE menunjukkan adanya kesalahan prediksi yang cukup besar, sementara nilai R² yang negatif menandakan bahwa model tidak mampu menjelaskan variabilitas data dengan baik. Hasil ini menunjukkan bahwa model perlu ditinjau ulang dan mungkin membutuhkan perbaikan signifikan, seperti pengoptimalan hyperparameter, perubahan pada preprocessing data, atau bahkan mencoba model yang berbeda.")
        elif button and option1:
            X = df2.drop(['harga_satuan'], axis=1)
            X = X.drop([option1], axis=1)
            y = df2[option1]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            
            test = X_test.drop(['nama_tanaman'], axis=1)

            model = joblib.load('model_svr_stock.pkl')
            y_pred = model.predict(test)
            
            st.write('**Hasil Prediksi Stok pada bulan mendatang**')
            result = pd.DataFrame({'Nama Tanaman':X_test['nama_tanaman'], 'Actual': y_test, 'Predicted': y_pred})
            st.table(result)

            st.write('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_pred),4))
            st.write('Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred),4))
            st.write('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),4))
            st.write('Coefficient of determination:', round(metrics.r2_score(y_test, y_pred),4))
            st.write('')

            st.markdown("Analisis hasil metrik yang diberikan memberikan gambaran tentang seberapa baik model prediksi stok Tanaman menggunakan Support Vector Regression (SVR) dalam memprediksi stok. Berikut adalah penjelasan untuk setiap metrik:")
            st.markdown("1. **Mean Absolute Error (MAE):** MAE mengukur rata-rata absolut dari selisih antara nilai yang diprediksi dan nilai aktual. Dalam hal ini, nilai MAE sebesar 0.7656 menunjukkan bahwa rata-rata kesalahan prediksi adalah sekitar 0.77. Nilai ini cukup rendah, yang menunjukkan bahwa prediksi model cukup akurat dalam hal kesalahan absolut rata-rata. Ini berarti, secara rata-rata, model membuat kesalahan prediksi sekitar 0.77 unit dari nilai aktualnya.")
            st.markdown("2. **Mean Squared Error (MSE):** MSE mengukur rata-rata dari kuadrat selisih antara nilai yang diprediksi dan nilai aktual. Karena kesalahan dikalikan dengan dirinya sendiri, MSE memberikan penalti yang lebih besar untuk kesalahan yang lebih besar. Nilai MSE sebesar 0.6955 menunjukkan bahwa ada variasi kesalahan yang cukup kecil, namun kesalahan yang besar cenderung mendominasi. MSE yang rendah menunjukkan bahwa prediksi model cukup dekat dengan nilai sebenarnya, tetapi karena ini adalah kuadrat dari kesalahan, angka ini juga mencerminkan sensitivitas terhadap outlier (kesalahan prediksi yang sangat besar).")
            st.markdown("3. **Root Mean Squared Error (RMSE):** RMSE adalah akar kuadrat dari MSE dan kembali ke satuan asli dari variabel yang diprediksi. RMSE sering digunakan karena lebih mudah untuk diinterpretasikan dibandingkan dengan MSE, terutama karena berada pada skala yang sama dengan data asli. Nilai RMSE sebesar 0.834 menunjukkan bahwa kesalahan prediksi rata-rata model adalah sekitar 0.834 unit. Ini berarti prediksi model, secara rata-rata, menyimpang sekitar 0.834 unit dari nilai sebenarnya. Nilai ini cukup kecil dan menunjukkan bahwa model bekerja dengan baik.")
            st.markdown("4. **Coefficient of Determination (R-squared):** R² mengukur proporsi variabilitas dalam data yang bisa dijelaskan oleh model. Nilai R² berkisar antara 0 hingga 1, di mana nilai yang mendekati 1 menunjukkan model yang sangat baik dalam menjelaskan variabilitas data. Nilai R² sebesar 0.9398 menunjukkan bahwa model menjelaskan sekitar 93.98% dari variabilitas dalam data target. Ini berarti model memiliki kinerja yang sangat baik, dan hanya sekitar 6.02% dari variasi data yang tidak dapat dijelaskan oleh model.")
            st.markdown("Metrik-metrik ini menunjukkan bahwa model Support Vector Regression (SVR) Anda bekerja dengan sangat baik untuk tugas prediksi stok tanaman. Kesalahan prediksi rata-rata rendah, dan model mampu menjelaskan sebagian besar variabilitas dalam data. Dengan nilai MAE dan RMSE yang kecil serta nilai R² yang mendekati 1, sehingga model ini memberikan prediksi yang akurat dan dapat diandalkan.")