# IMPORT LIBRARIES
import pandas as pd
import time
import pickle
import joblib
import streamlit as st
import altair as alt
from streamlit import config as st_config
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler, LabelEncoder

# IMPORT DATASETS
df = pd.read_csv('./dataset/heartdisease_cleansed.csv')
X = df[['cp', 'thalach', 'slope', 'oldpeak', 'exang', 'ca', 'thal', 'sex', 'age']]
y = df['target']

# IMPORT MODEL
with open('./model_file/model.joblib', 'rb') as file:
    model = joblib.load(file)
model = model.best_estimator_


# Set page general info
def set_page_configuration():
    st.set_page_config(
        page_title=f'Heart Disease - Diagnose',
        page_icon='🫀',
        layout='wide',
        initial_sidebar_state='expanded')


# Menu 1
def menu_analysis():
    st.header('Insight and Analysis Process')
    tab1, tab2 = st.tabs(['Background', 'Visualization'])
    with tab1:
        st.subheader('Case Study')
        st.write('Cardiovascular disease (CVDs) atau penyakit jantung merupakan penyebab kematian nomor satu secara global '
                 'dengan 17,9 juta kasus kematian setiap tahunnya. Penyakit jantung disebabkan oleh hipertensi, obesitas, '
                 'dan gaya hidup yang tidak sehat. Deteksi dini penyakit jantung perlu dilakukan pada '
                 'kelompok risiko tinggi agar dapat segera mendapatkan penanganan dan pencegahan. '
                 'Sehingga tujuan bisnis yang ingin dicapai yaitu membentuk model prediksi penyakit jantung pada pasien '
                 'berdasarkan feature-feature yang ada untuk membantu para dokter melakukan diagnosa secara tepat dan '
                 'akurat. Harapannya agar penyakit jantung dapat ditangani lebih awal. Dengan demikian, '
                 'diharapkan juga angka kematian akibat penyakit jantung dapat turun.')
        st.divider()
        st.subheader('About Datasets')
        st.markdown('''
        Melalui dataset Heart Disease yang diakses dari UCI ML: https://archive.ics.uci.edu/dataset/45/heart+disease  
        Dataset yang digunakan ini berasal dari tahun 1988 dan terdiri dari empat database: Cleveland, Hungaria, Swiss, 
        dan Long Beach V.  
          
        Berikut keterangan mengenai atribut pada dataset:  
        1. age: variabel ini merepresentasikan usia pasien yang diukur dalam tahun.  
        2. sex: variabel ini merepresentasikan jenis kelamin pasien dengan nilai 1 untuk laki-laki dan nilai 0 untuk perempuan.  
        3. cp (Chest pain type): variabel ini merepresentasikan jenis nyeri dada yang dirasakan oleh pasien dengan 4 nilai kategori yang mungkin: nilai 1 mengindikasikan nyeri dada tipe angina, nilai 2 mengindikasikan nyeri dada tipe nyeri tidak stabil, nilai 3 mengindikasikan nyeri dada tipe nyeri tidak stabil yang parah, dan nilai 4 mengindikasikan nyeri dada yang tidak terkait dengan masalah jantung.  
        4. trestbps (Resting blood pressure): variabel ini merepresentasikan tekanan darah pasien pada saat istirahat, diukur dalam mmHg (milimeter air raksa (merkuri)).  
        5. chol (Serum cholestoral): variabel ini merepresentasikan kadar kolesterol serum dalam darah pasien, diukur dalam mg/dl (miligram per desiliter).  
        6. fbs (Fasting blood sugar): variabel ini merepresentasikan kadar gula darah pasien saat puasa (belum makan) dengan nilai 1 jika kadar gula darah > 120 mg/dl dan nilai 0 jika tidak.  
        7. restecg (Resting electrocardiographic results): variabel ini merepresentasikan hasil elektrokardiogram pasien saat istirahat dengan 3 nilai kategori yang mungkin: nilai 0 mengindikasikan hasil normal, nilai 1 mengindikasikan adanya kelainan gelombang ST-T, dan nilai 2 mengindikasikan hipertrofi ventrikel kiri.  
        8. thalach (Maximum heart rate achieved): variabel ini merepresentasikan detak jantung maksimum yang dicapai oleh pasien selama tes olahraga, diukur dalam bpm (denyut per menit).  
        9. exang (Exercise induced angina): variabel ini merepresentasikan apakah pasien mengalami angina (nyeri dada) yang dipicu oleh aktivitas olahraga, dengan nilai 1 jika ya dan nilai 0 jika tidak.  
        10. oldpeak: variabel ini merepresentasikan seberapa banyak ST segmen menurun atau depresi saat melakukan aktivitas fisik dibandingkan saat istirahat.  
        11. slope: variabel ini merepresentasikan kemiringan segmen ST pada elektrokardiogram (EKG) selama latihan fisik maksimal dengan 3 nilai kategori.  
        12. ca (Number of major vessels): variabel ini merepresentasikan jumlah pembuluh darah utama (0-3) yang terlihat pada pemeriksaan flourosopi.  
        13. thal: variabel ini merepresentasikan hasil tes thalium scan dengan 3 nilai kategori yang mungkin:  
        - thal 1: menunjukkan kondisi normal.  
        - thal 2: menunjukkan adanya defek tetap pada thalassemia.  
        - thal 3: menunjukkan adanya defek yang dapat dipulihkan pada thalassemia.  
        14. target: 0 = tidak ada penyakit dan 1 = penyakit.
        ''')

    with tab2:
        st.subheader('Metrics')
        st.markdown('''
        1. Jumlah Penderita penyakit jantung  
        2. Proporsi pasien pria dan wanita yang mengalami penyakit jantung  
        3. Usia rata rata pasien yang terkena penyakit jantung
        ''')
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            plot1 = alt.Chart(df, title='1. Jumlah Penderita CVDs').mark_bar().encode(
                alt.X('target:N', title='').axis(labelAngle=0),
                alt.Y('count()', title=''),
                alt.Color('target:N', legend=None)
            ).properties(width=500, height=460)
            st.altair_chart(plot1, use_container_width=False)
        with col2:
            plot2 = alt.Chart(df, title='2. Proporsi Pria dan Wanita').mark_bar().encode(
                alt.Column('target', title=''),
                alt.X('sex', title='').axis(labelAngle=0),
                alt.Y('count()', title=''),
                alt.Color('target', title='')
            ).properties(width=300)
            st.altair_chart(plot2, use_container_width=False)

        st.divider()
        option = st.radio('Pilih Gender', ['Semua Gender', 'Wanita', 'Pria'])
        dict_sex = {'Wanita':'Female', 'Pria':'Male'}
        if option == 'Semua Gender':
            df3 = df
        else:
            df3 = df[df['sex'] == dict_sex[option]]
            
        plot3 = alt.Chart(df3, title='3. Rentang Usia Penderita CVDs').mark_bar().encode(
            alt.X('age:N', title='Umur (Tahun)').bin().axis(labelAngle=0),
            alt.Y('count()', title='')
        )
        st.altair_chart(plot3, use_container_width=True)


# Menu 2
def menu_analytics():
    st.title('Diagnose Testing')
    flag = False

    tab1, tab2 = st.tabs(['Screening', 'Suggestion'])
    with tab1:
        with st.form('Input'):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader('Identity')
                iName = st.text_input('Enter Your Name', placeholder='your name...')
                iAge = st.number_input('Select Age',
                                       value=None,
                                       placeholder='Type your age...',
                                       min_value=0)
                iSex = st.radio('Select Gender',
                                ['Male', 'Female'])

            with col2:
                st.subheader('Physical Examination')
                iThalach = st.number_input('Maximum Heart Rate',
                                           value=None,
                                           placeholder='in bpm...',
                                           min_value=0)
                iOldpeak = st.number_input('ST Depression',
                                           value=None,
                                           placeholder='in mm...',
                                           step=0.1)
                if iOldpeak is not None and iOldpeak < 0:
                    st.write('⚠️ST Segment cannot less than 0 ⚠️')
                iSlope = st.selectbox('ST Segment Slope',
                                      ['Down Sloping', 'Flat', 'Up Sloping'],
                                      index=None,
                                      placeholder='select ST slope...')
                iThal = st.selectbox('Select Blood Disorder',
                                     ['Fixed Defect', 'Normal Blood Flow', 'Reversible Defect'],
                                     index=None,
                                     placeholder='thallium Scan...')
                iCa = st.slider('Number of Major Vessels', 0, 3)

            with col3:
                st.subheader('Symptoms')

                iCP = st.selectbox('Chest Pain Type',
                                   ['Asymptomatic', 'Atypical Angina', 'Non-Anginal Pain', 'Typical Angina'],
                                   index=None,
                                   placeholder='select CP type...')
                iExang = st.radio('Does Excercise Induced Angina?',
                                  ['Yes', 'No'])

            submitted = st.form_submit_button('Submit', use_container_width=True)

            dSlope = {key: value for key, value in zip(['Down Sloping', 'Flat', 'Up Sloping'],
                                                       ['downsloping', 'flat', 'upsloping'])}
            dThal = {key: value for key, value in zip(['Fixed Defect', 'Normal Blood Flow', 'Reversible Defect'],
                                                      ['fixed defect', 'normal', 'reversable defect'])}
            dCa = {key: value for key, value in zip([2, 0, 1, 3],
                                                    ['Number of major vessels: 2', 'Number of major vessels: 0',
                                                     'Number of major vessels: 1', 'Number of major vessels: 3'])}
            dCP = {key: value for key, value in
                   zip(['Asymptomatic', 'Atypical Angina', 'Non-Anginal Pain', 'Typical Angina'],
                       ['asymtomatic', 'atypical angina', 'non-anginal pain', 'typical angina'])}

            if (iCP is not None) or (iSlope is not None) or (iThal is not None):
                cp, slope, ca, thal = dCP[iCP], dSlope[iSlope], dCa[iCa], dThal[iThal]

                X.loc[len(X.index)] = [cp, iThalach, slope, iOldpeak, iExang, ca, thal, iSex, iAge]

                # Label Encoding
                cat_col = X.select_dtypes('object')
                encoder = {}
                for m in cat_col.columns.tolist():
                    label_encoder = LabelEncoder()
                    X[m] = label_encoder.fit_transform(X[m])
                    encoder[m] = label_encoder

                scaler = StandardScaler()
                features = scaler.fit_transform(X)
                target = model.predict_proba(features)
                target_no = target[len(target) - 1][0]
                target_yes = target[len(target) - 1][1]
                label = pd.DataFrame({'label': ['Diagnosed with Heart Disease'],
                                      'Yes': [target_yes],
                                      'No': [target_no]})

                if submitted:
                    with st.spinner('Please Wait...'):
                        time.sleep(1)

                    a, b = st.columns([1, 5])
                    with b:
                        b1, b2 = st.columns([1, 2.25])
                        b2.subheader('Prediction Result')

                        base = alt.Chart(label)
                        middle = base.encode(
                            y=alt.Y('label', axis=None),
                            text=alt.Text('label'),
                            color=alt.value('white')
                        ).mark_text().properties(width=175)

                        left = base.encode(
                            y=alt.Y('label', axis=None),
                            x=alt.X('Yes', title='', scale=alt.Scale(domain=[0, 1]), sort=alt.SortOrder('descending')).axis(format='%'),
                            color=alt.value('#E64242')
                        ).mark_bar().properties(title=alt.TitleParams(text='Yes', dx=380))

                        right = base.encode(
                            y=alt.Y('label', axis=None),
                            x=alt.X('No', title='', scale=alt.Scale(domain=[0, 1])).axis(format='%'),
                            color=alt.value('green')
                        ).mark_bar().properties(title='No')

                        pred = alt.hconcat(left, middle, right, spacing=5)
                        st.altair_chart(pred, use_container_width=True)

                        if target_yes >= target_no:
                            st.markdown(f'''
                            Hi {iName} ! from screening test, you're potentially suffered heart disease.  
                            You have {target_yes * 100:.2f}% chance diagnosed with heart disease.
                            ''')
                        else:
                            st.markdown(f'''
                            Hi {iName} ! from screening test, you're not potentially suffered heart disease.  
                            You have {target_yes * 100:.2f}% chance diagnosed with heart disease.
                            ''')

                    submitted = False
                    flag = True

    with tab2:
        if flag:
            st.write('Based on the prediction result, here are some action that might help you:')

            if target_yes >= target_no:
                st.subheader('Preventive Action')
                st.markdown('**---**')
                st.subheader('Reactive Action')
                st.markdown('''
                1. **Consult to Cardiologist:** The first and most crucial step is to seek cardiologist. 
                They can provide a comprehensive evaluation of the individual's condition, 
                recommend treatments, and create a tailored care plan.  
                2. **Medication Adherence:** If prescribed medications, it's crucial to take them as directed by the 
                healthcare provider. Skipping or altering doses can negatively impact heart health.  
                3. **Continuous Monitoring:** Keep track of Cardiovascular and 
                follow any prescribed medication or lifestyle changes to maintain healthy Cardiovascular.  
                4. **Emergency Plan:** Develop a plan for handling emergencies, 
                such as knowing the signs of a heart attack and what to do if one occurs.
                ''')
            else:
                st.subheader('Preventive Action')
                st.markdown('''
                1. **Regular Exercise:** Engage in regular physical activity as recommended by the cardiologist. 
                Exercise can improve heart health, reduce stress, and help with weight management.  
                2. **Regular Check-ups:** Schedule regular follow-up appointments with the cardiologist to 
                monitor the progress of the condition and make necessary adjustments to the treatment plan.  
                3. **Smoking Cessation:** Quitting smoke is one of the most effective ways to improve heart health 
                even if you're diagnosed with any particular cardiac disorder.  
                4. **Limit Alcohol Intake:** Limit alcohol consumption to moderate levels, 
                as excessive drinking can have a negative impact on heart health.  
                5. **Education:** Educate oneself about heart disease, its risk factors, 
                and the importance of managing the condition. 
                Knowledge empowers individuals to make informed decisions about their health.
                ''')
                st.subheader('Reactive Action')
                st.markdown('**---**')
        else:
            st.subheader('Please Have the Examination First!')


# Navigation Menu
def navbar_menu():
    with st.sidebar:
        selected_navbar_menu = option_menu(
            menu_title='Menu',
            options=['Insight', 'Diagnose'],
            icons=['graph-up-arrow', 'lightbulb-fill'],
            menu_icon='cast',
            default_index=0,
            orientation='vertical',
            styles={'nav-link': {'--hover-color': '#FFB0B0',
                                 '--active-background-color': '#E64242'},
                    'nav-link-selected': {'background-color': '#E64242'},
                    })

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown("""
                    Developed by:    
                    - Fery Kurniawan
                    """)
        st.caption('Capstone Project: ML and AI Batch 7')

        return selected_navbar_menu


def main():
    st_config.set_option('server.maxUploadSize', 1000)

    # display_title()
    selected_navbar_menu = navbar_menu()
    if selected_navbar_menu == 'Insight':
        menu_analysis()
    if selected_navbar_menu == 'Diagnose':
        menu_analytics()


# Run Program
if __name__ == '__main__':
    set_page_configuration()
    main()
