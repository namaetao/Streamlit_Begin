import streamlit as st
import pandas as pd
import joblib
import os

# ตรวจสอบว่าโมเดลถูกบันทึกหรือไม่
model_path = 'model/sale_regression_model.pkl'
if not os.path.exists(model_path):
    st.error(f"Error: Model file '{model_path}' not found. Please ensure the model is saved correctly in Step 7.")
else:
    # โหลดโมเดลที่บันทึกไว้
    loaded_model = joblib.load(model_path)

    st.title('Sales Prediction Web App 📈')
    st.write('กรุณากรอกงบประมาณการโฆษณาในแต่ละช่องทางเพื่อพยากรณ์ยอดขาย')

    # สร้างช่องกรอกข้อมูลสำหรับผู้ใช้
    tv_budget = st.number_input('งบประมาณ TV (ล้านบาท)', min_value=0.0, max_value=300.0, value=150.0, step=1.0)
    radio_budget = st.number_input('งบประมาณ Radio (ล้านบาท)', min_value=0.0, max_value=50.0, value=20.0, step=0.1)
    newspaper_budget = st.number_input('งบประมาณ Newspaper (ล้านบาท)', min_value=0.0, max_value=100.0, value=10.0, step=0.1)

    # ปุ่มสำหรับพยากรณ์
    if st.button('พยากรณ์ยอดขาย'):
        # สร้าง DataFrame จากข้อมูลที่ผู้ใช้กรอก
        unseen_data_for_prediction = pd.DataFrame({
            'TV': [tv_budget],
            'Radio': [radio_budget],
            'Newspaper': [newspaper_budget]
        })

        # ทำการพยากรณ์
        predicted_sales = loaded_model.predict(unseen_data_for_prediction)[0]

        st.success(f'ยอดขายที่คาดการณ์: {predicted_sales:,.2f} ล้านบาท')

