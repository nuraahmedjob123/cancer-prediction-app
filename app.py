import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# قائمة بأسماء الأعمدة (تمت إضافتها هنا)
column_names = [
    "Sample_code_number", "Clump_Thickness", "Uniformity_of_Cell_Size",
    "Uniformity_of_Cell_Shape", "Marginal_Adhesion", "Single_Epithelial_Cell_Size",
    "Bare_Nuclei", "Bland_Chromatin", "Normal_Nucleoli", "Mitoses", "Class",
]

# تعيين إعدادات الصفحة
st.set_page_config(
    page_title="تطبيق التنبؤ بالسرطان",
    layout="wide",
    initial_sidebar_state="expanded",
)

# تحميل النموذج المحفوظ
try:
    model = joblib.load('cancer_prediction_model.pkl')
except FileNotFoundError:
    st.error("لم يتم العثور على ملف النموذج 'cancer_prediction_model.pkl'. يرجى التأكد من وجوده في نفس المجلد.")

# دالة لرسم حدود قرار النموذج
def plot_decision_boundary(model, X, y):
    # نختار أهم ميزتين للرسم
    feature1_name = "Uniformity_of_Cell_Size"
    feature2_name = "Clump_Thickness"
    
    # نحصل على قيم الميزتين من البيانات الأصلية
    X_plot = X[[feature1_name, feature2_name]]
    y_plot = y

    x_min, x_max = X_plot.iloc[:, 0].min() - 1, X_plot.iloc[:, 0].max() + 1
    y_min, y_max = X_plot.iloc[:, 1].min() - 1, X_plot.iloc[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    scaler = model.named_steps['scaler']
    Z_input = np.c_[xx.ravel(), yy.ravel()]
    dummy_features = np.zeros((Z_input.shape[0], 7))
    Z_full_input = np.concatenate([Z_input, dummy_features], axis=1) # تعديل هذا السطر
    
    Z = model.predict(Z_full_input)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    plt.scatter(X_plot.iloc[:, 0], X_plot.iloc[:, 1], c=y_plot, s=20, edgecolor='k', cmap='viridis')
    plt.xlabel(f"({feature1_name}) انتظام حجم الخلية")
    plt.ylabel(f"({feature2_name}) سمك الكتلة")
    plt.title("حدود قرار النموذج")
    plt.legend(['حميد', 'خبيث'], loc='upper right')
    return plt

# اختيار الصفحة من الشريط الجانبي
st.sidebar.title("قائمة التنقل")
page = st.sidebar.selectbox("اختر الصفحة", ["الرئيسية", "التنبؤ"])

# --- واجهة الترحيب ---
if page == "الرئيسية":
    st.header("تطبيق التنبؤ بالسرطان 🔬")
    st.image("https://images.unsplash.com/photo-1549487779-111af34c9c1b?q=80&w=1470&auto=format&fit=crop")
    st.markdown("---")
    
    st.container()
    st.write("""
        ### مرحبًا بكم في هذا التطبيق المبتكر
        هذا التطبيق هو أداة ذكية مبنية على التعلم الآلي للتنبؤ بنوع الورم السرطاني (حميد أو خبيث) بناءً على خصائص معينة للخلايا.
        
        تم بناء النموذج باستخدام بيانات من موقع UCI، ليكون قادرًا على مساعدة الأطباء والباحثين في تقييم الحالات بشكل أسرع وأكثر دقة.
    """)
    st.markdown("---")
    
    st.subheader("عن المالك")
    st.info("تم تطوير هذا التطبيق بواسطة: **نورا الحرازي**")
    
    st.markdown("---")
    st.write("للبدء، يرجى اختيار **'التنبؤ'** من القائمة الجانبية.")

# --- واجهة إدخال البيانات والتنبؤ ---
elif page == "التنبؤ":
    st.header("صفحة التنبؤ 📊")
    st.markdown("---")
    
    with st.expander("❤️ رسائل ونصائح إيجابية ❤️", expanded=True):
        st.markdown("""
            * **القوة في المعرفة:** هذا التطبيق يمنحك أداة لفهم البيانات، لكن تذكر دائمًا أنه ليس بديلاً عن استشارة الطبيب المختص.
            * **الأمل دائمًا موجود:** التقدم في مجال الطب يفتح آفاقًا جديدة للعلاج.
        """)

    st.markdown("---")
    st.write("أدخل خصائص العينة للتنبؤ بنوع الورم.")

    col1, col2 = st.columns(2)
    with col1:
        clump_thickness = st.slider("سمك الكتلة", 1, 10, 5)
        cell_size = st.slider("انتظام حجم الخلية", 1, 10, 5)
        cell_shape = st.slider("انتظام شكل الخلية", 1, 10, 5)
        marginal_adhesion = st.slider("الالتصاق الهامشي", 1, 10, 5)
        single_epithelial_cell_size = st.slider("حجم الخلية الظهارية", 1, 10, 5)
    with col2:
        bare_nuclei = st.slider("النواة العارية", 1, 10, 5)
        bland_chromatin = st.slider("الكروماتين الأملس", 1, 10, 5)
        normal_nucleoli = st.slider("النويّات الطبيعية", 1, 10, 5)
        mitoses = st.slider("الانقسام الفتيلي", 1, 10, 5)

    st.markdown("---")
    if st.button("التنبؤ"):
        if 'model' in locals():
            user_input = np.array([[clump_thickness, cell_size, cell_shape, marginal_adhesion,
                                   single_epithelial_cell_size, bare_nuclei, bland_chromatin,
                                   normal_nucleoli, mitoses]])
            columns = ["Clump_Thickness", "Uniformity_of_Cell_Size", "Uniformity_of_Cell_Shape",
                       "Marginal_Adhesion", "Single_Epithelial_Cell_Size", "Bare_Nuclei",
                       "Bland_Chromatin", "Normal_Nucleoli", "Mitoses"]
            input_df = pd.DataFrame(user_input, columns=columns)
            prediction = model.predict(input_df)

            result_col, icon_col = st.columns([4, 1])
            with result_col:
                st.write("### النتيجة:")
                if prediction[0] == 1:
                    st.error("خبيث (Malignant)")
                else:
                    st.success("حميد (Benign)")
            
            with icon_col:
                if prediction[0] == 1:
                    st.image("https://cdn-icons-png.flaticon.com/512/189/189680.png", width=50)
                else:
                    st.image("https://cdn-icons-png.flaticon.com/512/3673/3673516.png", width=50)
            
            st.markdown("---")
            st.write("البيانات المدخلة:")
            st.table(input_df)

            # --- عرض المخطط ---
            st.markdown("---")
            st.subheader("تصور حدود قرار النموذج")
            
            # نحتاج إلى تحميل البيانات الأصلية مرة أخرى لرسم المخطط
            data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
            data_df = pd.read_csv(data_url, names=column_names, na_values="?")
            data_df.dropna(inplace=True)
            data_df['Class'] = data_df['Class'].map({2: 0, 4: 1})
            data_df['Bare_Nuclei'] = pd.to_numeric(data_df['Bare_Nuclei'])
            
            X_full = data_df.drop(['Sample_code_number', 'Class'], axis=1)
            y_full = data_df['Class']

            fig = plot_decision_boundary(model, X_full, y_full)
            st.pyplot(fig)